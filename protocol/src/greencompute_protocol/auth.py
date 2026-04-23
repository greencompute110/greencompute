"""Request signing and verification for miner ↔ validator communication.

Supports two modes:
- **hotkey** (production): Miners sign with their ed25519 hotkey private key.
  Validators verify using the public key from the on-chain metagraph.
  No shared secret needed — fully decentralized.
- **hmac** (local dev): Classic HMAC-SHA256 with a shared secret.
  Used when no hotkey keypair is available.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from time import time
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SignedRequest(BaseModel):
    actor_id: str  # hotkey SS58 address
    nonce: str = Field(default_factory=lambda: secrets.token_hex(8))
    timestamp: int = Field(default_factory=lambda: int(time()))
    signature: str
    auth_mode: str = "hmac"  # "hotkey" or "hmac"


class ReplayStore(Protocol):
    def mark_seen(self, actor_id: str, nonce: str, timestamp: int) -> bool: ...


class MemoryReplayStore:
    def __init__(self) -> None:
        self._seen: set[tuple[str, str, int]] = set()

    def mark_seen(self, actor_id: str, nonce: str, timestamp: int) -> bool:
        key = (actor_id, nonce, timestamp)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    reason: str | None = None


def _canonical(actor_id: str, nonce: str, timestamp: int, body: bytes) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return f"{actor_id}:{nonce}:{timestamp}:{digest}".encode()


# ---------------------------------------------------------------------------
# HMAC signing (local dev / backward compat)
# ---------------------------------------------------------------------------


def sign_payload(secret: str, actor_id: str, body: bytes, nonce: str | None = None) -> SignedRequest:
    """Sign with HMAC-SHA256 shared secret (local dev mode)."""
    request = SignedRequest(
        actor_id=actor_id,
        nonce=nonce or secrets.token_hex(8),
        signature="",
        auth_mode="hmac",
    )
    signature = hmac.new(
        secret.encode(),
        _canonical(request.actor_id, request.nonce, request.timestamp, body),
        hashlib.sha256,
    ).hexdigest()
    return request.model_copy(update={"signature": signature})


def verify_payload(
    secret: str,
    signed: SignedRequest,
    body: bytes,
    replay_store: ReplayStore,
    now: int | None = None,
    window_seconds: int = 60,
) -> VerificationResult:
    """Verify HMAC-SHA256 signed request (local dev mode)."""
    current_time = now or int(time())
    if abs(current_time - signed.timestamp) > window_seconds:
        return VerificationResult(valid=False, reason="signature expired")
    if not replay_store.mark_seen(signed.actor_id, signed.nonce, signed.timestamp):
        return VerificationResult(valid=False, reason="replay detected")
    expected = hmac.new(
        secret.encode(),
        _canonical(signed.actor_id, signed.nonce, signed.timestamp, body),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signed.signature):
        return VerificationResult(valid=False, reason="signature mismatch")
    return VerificationResult(valid=True)


# ---------------------------------------------------------------------------
# Ed25519 hotkey signing (production / Bittensor)
# ---------------------------------------------------------------------------


def load_hotkey_from_wallet(coldkey_name: str, hotkey_name: str = "default") -> str:
    """Load hotkey seed from Bittensor wallet.

    Reads from ~/.bittensor/wallets/{coldkey_name}/hotkeys/{hotkey_name}.

    Returns:
        Seed hex string for Keypair.create_from_seed().
    """
    import json
    import os

    wallet_path = os.path.expanduser(
        f"~/.bittensor/wallets/{coldkey_name}/hotkeys/{hotkey_name}"
    )
    try:
        with open(wallet_path, "r") as f:
            wallet = json.load(f)
        seed = wallet.get("secretSeed") or wallet.get("privateKey") or wallet.get("seed") or wallet.get("private_key")
        if not seed:
            raise ValueError(f"No secretSeed/privateKey found in {wallet_path}")
        if isinstance(seed, str) and seed.startswith("0x"):
            seed = seed[2:]
        return seed
    except Exception as exc:
        raise RuntimeError(f"Failed to load hotkey from {wallet_path}: {exc}") from exc


def sign_payload_hotkey(hotkey_seed: str, actor_id: str, body: bytes, nonce: str | None = None) -> SignedRequest:
    """Sign with ed25519 hotkey private key (production mode).

    Args:
        hotkey_seed: Seed hex (loaded from Bittensor wallet file).
                     Use load_hotkey_from_wallet() to load from ~/.bittensor/wallets/{coldkey}/hotkeys/{hotkey}.
        actor_id: The hotkey SS58 address.
        body: Request body bytes.
    """
    from substrateinterface import Keypair

    request = SignedRequest(
        actor_id=actor_id,
        nonce=nonce or secrets.token_hex(8),
        signature="",
        auth_mode="hotkey",
    )
    message = _canonical(request.actor_id, request.nonce, request.timestamp, body)
    keypair = Keypair.create_from_seed(hotkey_seed)
    sig = keypair.sign(message)
    return request.model_copy(update={"signature": sig.hex()})


def verify_payload_hotkey(
    signed: SignedRequest,
    body: bytes,
    replay_store: ReplayStore,
    now: int | None = None,
    window_seconds: int = 60,
) -> VerificationResult:
    """Verify ed25519 signature using the hotkey's public key (production mode).

    No shared secret — the validator only needs the miner's SS58 address
    (hotkey), which it gets from the metagraph.
    """
    from substrateinterface import Keypair

    current_time = now or int(time())
    if abs(current_time - signed.timestamp) > window_seconds:
        return VerificationResult(valid=False, reason="signature expired")
    if not replay_store.mark_seen(signed.actor_id, signed.nonce, signed.timestamp):
        return VerificationResult(valid=False, reason="replay detected")

    message = _canonical(signed.actor_id, signed.nonce, signed.timestamp, body)
    try:
        sig_bytes = bytes.fromhex(signed.signature)
    except ValueError:
        return VerificationResult(valid=False, reason="invalid signature encoding")

    try:
        keypair = Keypair(ss58_address=signed.actor_id)
        if not keypair.verify(message, sig_bytes):
            return VerificationResult(valid=False, reason="signature mismatch")
    except Exception as exc:
        logger.debug("hotkey verification error: %s", exc)
        return VerificationResult(valid=False, reason="signature verification failed")

    return VerificationResult(valid=True)
