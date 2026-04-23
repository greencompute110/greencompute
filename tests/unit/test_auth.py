from greencompute_protocol.auth import MemoryReplayStore, sign_payload, verify_payload


def test_signing_and_replay_protection():
    body = b'{"hello":"world"}'
    replay_store = MemoryReplayStore()
    signed = sign_payload(secret="top-secret", actor_id="miner-a", body=body)

    valid = verify_payload("top-secret", signed, body, replay_store)
    replay = verify_payload("top-secret", signed, body, replay_store)
    expired = verify_payload("top-secret", signed, body, MemoryReplayStore(), now=signed.timestamp + 120)

    assert valid.valid is True
    assert replay.valid is False
    assert replay.reason == "replay detected"
    assert expired.valid is False
    assert expired.reason == "signature expired"

