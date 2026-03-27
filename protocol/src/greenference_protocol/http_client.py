"""HTTP client for remote control-plane communication.

Replaces in-process ControlPlaneService imports, allowing agents to talk to a
remote control-plane over HTTP with signed requests.

Supports two auth modes:
- **hotkey**: ed25519 signing with the miner's hotkey keypair (production)
- **hmac**: HMAC-SHA256 with a shared secret (local dev)
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from greenference_protocol.auth import load_hotkey_from_wallet, sign_payload, sign_payload_hotkey
from greenference_protocol.models import (
    CapacityUpdate,
    DeploymentRecord,
    DeploymentStatusUpdate,
    Heartbeat,
    LeaseAssignment,
    MinerRegistration,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)


class ControlPlaneHTTPError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class ControlPlaneHTTPClient:
    """Signed HTTP client for the control-plane miner API.

    Args:
        base_url: Control-plane base URL.
        hotkey: Miner's SS58 hotkey address.
        auth_secret: HMAC shared secret (used when auth_mode="hmac").
        hotkey_wallet_path: Path to Bittensor hotkey file (~/.bittensor/wallets/{coldkey}/hotkeys/{hotkey}).
                           Used for ed25519 signing in production (auth_mode="hotkey").
        auth_mode: "hotkey" for ed25519 (production) or "hmac" for shared secret (dev).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        hotkey: str,
        auth_secret: str = "",
        hotkey_wallet_path: str | None = None,
        auth_mode: str = "hmac",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hotkey = hotkey
        self.auth_secret = auth_secret
        self.auth_mode = auth_mode
        self.timeout = timeout

        # Load hotkey seed from wallet file in production mode
        self.hotkey_seed: str | None = None
        if auth_mode == "hotkey" and hotkey_wallet_path:
            self.hotkey_seed = load_hotkey_from_wallet(hotkey_wallet_path)

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        if self.auth_mode == "hotkey" and self.hotkey_seed:
            signed = sign_payload_hotkey(self.hotkey_seed, self.hotkey, body)
        else:
            signed = sign_payload(self.auth_secret, self.hotkey, body)
        return {
            "X-Miner-Hotkey": signed.actor_id,
            "X-Miner-Signature": signed.signature,
            "X-Miner-Nonce": signed.nonce,
            "X-Miner-Timestamp": str(signed.timestamp),
            "X-Miner-Auth-Mode": signed.auth_mode,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, body: bytes | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = body if body else b""
        headers = self._signed_headers(data)
        req = Request(url, data=data if method != "GET" else None, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            detail = exc.read().decode() if exc.fp else str(exc)
            logger.error("control-plane %s %s -> %s: %s", method, path, exc.code, detail)
            raise ControlPlaneHTTPError(exc.code, detail) from exc

    def _post(self, path: str, payload: Any) -> Any:
        body = payload.model_dump_json().encode() if hasattr(payload, "model_dump_json") else json.dumps(payload).encode()
        return self._request("POST", path, body)

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    # --- Miner lifecycle ---

    def register_miner(self, registration: MinerRegistration) -> MinerRegistration:
        data = self._post("/miner/v1/register", registration)
        return MinerRegistration.model_validate(data)

    def record_heartbeat(self, heartbeat: Heartbeat) -> Heartbeat:
        data = self._post("/miner/v1/heartbeat", heartbeat)
        return Heartbeat.model_validate(data)

    def update_capacity(self, update: CapacityUpdate) -> CapacityUpdate:
        data = self._post("/miner/v1/capacity", update)
        return CapacityUpdate.model_validate(data)

    def list_leases(self, hotkey: str) -> list[LeaseAssignment]:
        data = self._get(f"/miner/v1/leases/{hotkey}")
        return [LeaseAssignment.model_validate(item) for item in data]

    def update_deployment_status(self, update: DeploymentStatusUpdate) -> dict:
        return self._post(f"/miner/v1/deployments/{update.deployment_id}/status", update)

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        try:
            data = self._get(f"/miner/v1/deployments/{deployment_id}")
            return DeploymentRecord.model_validate(data)
        except ControlPlaneHTTPError as exc:
            if exc.status == 404:
                return None
            raise

    def get_workload(self, workload_id: str) -> WorkloadSpec | None:
        try:
            data = self._get(f"/miner/v1/workloads/{workload_id}")
            return WorkloadSpec.model_validate(data)
        except ControlPlaneHTTPError as exc:
            if exc.status == 404:
                return None
            raise
