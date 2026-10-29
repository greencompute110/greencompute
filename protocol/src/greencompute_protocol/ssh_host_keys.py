"""A pod's SSH host keys, as the node reads them and renters check them.

A pod's sshd shows its host key on every connection. Host ports are reused
from one rental to the next, so a client that remembers keys by host and port
sees "host key changed" when a new pod lands on a port it has used before,
and the usual workaround is to turn host-key checking off. Publishing each
pod's keys lets the client check them instead (external API user, request
#9, Oct 2026).

The keys come from the machine, so only well-formed public keys get through:
a known type, base64 that decodes, and a key blob that names that same type.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import struct
from collections.abc import Iterable

KEY_TYPES = frozenset({
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
})
MAX_KEYS = 8
# An 8192-bit RSA key is about 1,400 base64 characters.
_MAX_KEY_CHARS = 2048


def _well_formed(key_type: str, encoded: str) -> bool:
    if key_type not in KEY_TYPES or len(encoded) > _MAX_KEY_CHARS:
        return False
    try:
        blob = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(blob) < 4:
        return False
    (length,) = struct.unpack(">I", blob[:4])
    return blob[4:4 + length] == key_type.encode()


def clean_ssh_host_keys(values: str | Iterable[object] | None) -> list[str]:
    """Each well-formed key in `values` as `type base64`: the lines of
    `ssh_host_*_key.pub` files, or keys already in that form. Comments are
    dropped (they name the container), repeats removed, at most MAX_KEYS."""
    if values is None:
        return []
    if isinstance(values, str):
        values = values.splitlines()
    out: list[str] = []
    try:
        items = list(values)
    except TypeError:
        return []
    for value in items:
        if not isinstance(value, str):
            continue
        parts = value.split()
        if len(parts) < 2 or not _well_formed(parts[0], parts[1]):
            continue
        key = f"{parts[0]} {parts[1]}"
        if key not in out:
            out.append(key)
        if len(out) >= MAX_KEYS:
            break
    return out


def ssh_key_fingerprint(key: str) -> str:
    """The SHA256 fingerprint `ssh` prints when it first meets the host,
    e.g. `SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s`."""
    blob = base64.b64decode(key.split()[1])
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def known_hosts_line(host: str, port: int, key: str) -> str:
    """The `~/.ssh/known_hosts` line for `key` at host:port, in the form ssh
    writes itself (`[host]:port` unless the port is 22)."""
    target = host if port == 22 else f"[{host}]:{port}"
    return f"{target} {key}"
