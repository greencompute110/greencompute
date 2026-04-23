from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import click
import pytest

from greencompute import client as client_module
from greencompute.cli import main


class _FakeResponse:
    def __init__(self, payload: dict | list | str, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        if isinstance(self.payload, str):
            return self.payload.encode()
        return json.dumps(self.payload).encode()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _fake_urlopen(target, timeout=None):  # type: ignore[no-untyped-def]
    if isinstance(target, str):
        if target.endswith("/platform/workloads"):
            return _FakeResponse([{"workload_id": "wl-1", "name": "demo"}])
        raise HTTPError(target, 404, "not found", hdrs=None, fp=None)

    path = target.full_url
    method = target.get_method()
    payload = json.loads(target.data.decode()) if target.data else {}

    if path.endswith("/platform/register"):
        return _FakeResponse(
            {
                "user_id": "user-1",
                "username": payload["username"],
                "email": payload.get("email"),
            }
        )
    if path.endswith("/platform/api-keys"):
        return _FakeResponse(
            {
                "key_id": "key-1",
                "name": payload["name"],
                "user_id": payload.get("user_id"),
                "secret": "gk_demo",
            }
        )
    if path.endswith("/platform/images/contexts"):
        return _FakeResponse(
            {
                "context_uri": "file:///tmp/greencompute-build-context.zip",
                "archive_name": payload["context_archive_name"],
                "size_bytes": 128,
            }
        )
    if path.endswith("/platform/images") and method == "GET":
        return _FakeResponse(
            [
                {
                    "build_id": "build-1",
                    "image": "demo/echo:latest",
                    "status": "published",
                }
            ]
        )
    if path.endswith("/platform/images"):
        return _FakeResponse(
            {
                "build_id": "build-1",
                "image": payload["image"],
                "status": "published",
            }
        )
    if path.endswith("/platform/builds") and method == "GET":
        return _FakeResponse(
            [
                {
                    "build_id": "build-1",
                    "image": "demo/echo:latest",
                    "status": "published",
                    "executor_name": "remote-http-builder",
                }
            ]
        )
    if path.endswith("/platform/builds/build-1") and method == "GET":
        return _FakeResponse(
            {
                "build_id": "build-1",
                "image": "demo/echo:latest",
                "status": "published",
                "executor_name": "remote-http-builder",
            }
        )
    if "/platform/builds/" in path and "/logs/stream" in path and method == "GET":
        return _FakeResponse(
            'data: {"stage":"staging","message":"staged context"}\n\n'
            'data: {"stage":"building","message":"remote build submitted"}\n\n'
            'data: {"status":"published","build_id":"build-1"}\n'
        )
    if "/platform/images/" in path and path.endswith("/history"):
        return _FakeResponse(
            [
                {
                    "build_id": "build-1",
                    "image": "demo/echo:latest",
                    "status": "published",
                }
            ]
        )
    if path.endswith("/platform/workloads") and method == "GET":
        return _FakeResponse([{"workload_id": "wl-1", "name": "demo", "image": "demo/echo:latest"}])
    if path.endswith("/platform/workloads"):
        return _FakeResponse(
            {
                "workload_id": "wl-1",
                "name": payload["name"],
                "image": payload["image"],
                "public": payload.get("public", False),
            }
        )
    if path.endswith("/platform/workloads/wl-1") and method == "PATCH":
        return _FakeResponse(
            {
                "workload_id": "wl-1",
                "name": "demo",
                "display_name": payload.get("display_name", "demo"),
                "readme": payload.get("readme"),
                "logo_uri": payload.get("logo_uri"),
                "tags": payload.get("tags", ["existing"]),
                "workload_alias": None if payload.get("clear_workload_alias") else payload.get("workload_alias", "demo"),
                "pricing_class": payload.get("pricing_class", "standard"),
                "public": payload.get("public", False),
            }
        )
    if path.endswith("/platform/workloads/wl-1/utilization") and method == "GET":
        return _FakeResponse(
            {
                "workload_id": "wl-1",
                "active_deployments": 1,
                "total_request_count": 42,
                "total_compute_seconds": 12.5,
            }
        )
    if path.endswith("/platform/workloads/wl-1/shares") and method == "POST":
        return _FakeResponse(
            {
                "share_id": "share-1",
                "workload_id": "wl-1",
                "shared_with_user_id": payload["shared_with_user_id"],
                "permission": payload.get("permission", "invoke"),
            }
        )
    if path.endswith("/platform/workloads/wl-1/shares") and method == "GET":
        return _FakeResponse(
            [
                {
                    "share_id": "share-1",
                    "workload_id": "wl-1",
                    "shared_with_user_id": "user-2",
                    "permission": "invoke",
                }
            ]
        )
    if path.endswith("/platform/workloads/wl-1/warmup") and method == "GET":
        return _FakeResponse(
            'data: {"workload_id":"wl-1","status":"warmup_started"}\n\ndata: {"workload_id":"wl-1","status":"warmup_complete"}\n'
        )
    if path.endswith("/platform/deployments"):
        if method == "GET":
            return _FakeResponse(
                [
                    {
                        "deployment_id": "dep-1",
                        "workload_id": "wl-1",
                        "state": "ready",
                        "requested_instances": 1,
                        "endpoint": "http://miner.local/v1/chat/completions",
                    }
                ]
            )
        return _FakeResponse(
            {
                "deployment_id": "dep-1",
                "workload_id": payload["workload_id"],
                "state": "scheduled",
            }
        )
    if path.endswith("/platform/deployments/dep-1") and method == "PATCH":
        return _FakeResponse(
            {
                "deployment_id": "dep-1",
                "workload_id": "wl-1",
                "state": "ready",
                "requested_instances": payload.get("requested_instances", 1),
                "fee_acknowledged": payload.get("fee_acknowledged", True),
            }
        )
    if path.endswith("/platform/deployments/dep-1") and method == "GET":
        return _FakeResponse(
            {
                "deployment_id": "dep-1",
                "workload_id": "wl-1",
                "state": "ready",
                "requested_instances": 1,
            }
        )
    if path.endswith("/v1/chat/completions"):
        if payload.get("stream"):
            return _FakeResponse('data: {"content":"greencompute-response: hi"}\n\ndata: [DONE]\n')
        return _FakeResponse(
            {
                "id": "resp-1",
                "model": payload["model"],
                "content": "greencompute-response: hi",
                "deployment_id": "dep-1",
            }
        )

    raise HTTPError(path, 404, "not found", hdrs=None, fp=None)


@pytest.mark.usefixtures("monkeypatch")
def test_cli_happy_path_against_local_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(client_module.request, "urlopen", _fake_urlopen)
    base_url = "http://greenference.test"
    module_root = tmp_path / "sdk"
    module_root.mkdir()
    monkeypatch.chdir(module_root)
    workload_file = module_root / "cli_workload.py"
    data_file = module_root / "cli_data.txt"
    data_file.write_text("hello", encoding="utf-8")
    workload_file.write_text(
        """
from greencompute import Image, NodeSelector, Workload

image = (
    Image(username="demo", name="echo", tag="latest")
    .from_base("python:3.12-slim")
    .add("cli_data.txt", "/app/cli_data.txt")
    .run_command("echo build")
)

workload = Workload(
    name="echo-model",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
    model_identifier="demo/echo-model",
)
""",
        encoding="utf-8",
    )
    module_ref = f"{workload_file}:workload"

    monkeypatch.setenv("GREENCOMPUTE_API_URL", base_url)
    commands = [
        ["greencompute", "--base-url", base_url, "register", "--username", "alice", "--email", "alice@example.com"],
        ["greencompute", "--base-url", base_url, "keys", "create", "--name", "default", "--user-id", "user-1"],
        ["greencompute", "--base-url", base_url, "build", module_ref],
        ["greencompute", "--base-url", base_url, "workloads", "create", module_ref, "--public"],
        ["greencompute", "--base-url", base_url, "images", "history", "demo/echo:latest"],
        ["greencompute", "--base-url", base_url, "builds", "list"],
        ["greencompute", "--base-url", base_url, "builds", "get", "build-1"],
        ["greencompute", "--base-url", base_url, "builds", "logs", "build-1"],
        ["greencompute", "--base-url", base_url, "builds", "wait", "build-1"],
        ["greencompute", "--base-url", base_url, "deploy", module_ref, "--accept-fee", "--wait"],
        ["greencompute", "--base-url", base_url, "deployments", "list"],
        ["greencompute", "--base-url", base_url, "deployments", "get", "dep-1"],
        ["greencompute", "--base-url", base_url, "deployments", "update", "dep-1", "--requested-instances", "2"],
        ["greencompute", "--base-url", base_url, "deployments", "wait", "dep-1"],
        [
            "greencompute",
            "--base-url",
            base_url,
            "workloads",
            "update",
            "wl-1",
            "--display-name",
            "Updated",
            "--readme",
            "README",
            "--logo-uri",
            "https://example.com/logo.png",
            "--tag",
            "llm",
            "--tag",
            "chat",
            "--clear-workload-alias",
            "--pricing-class",
            "premium",
            "--public",
        ],
        ["greencompute", "--base-url", base_url, "workloads", "utilization", "wl-1"],
        ["greencompute", "--base-url", base_url, "workloads", "share", "wl-1", "--user-id", "user-2"],
        ["greencompute", "--base-url", base_url, "workloads", "shares", "wl-1"],
        ["greencompute", "--base-url", base_url, "workloads", "warmup", "wl-1"],
        ["greencompute", "--base-url", base_url, "run", module_ref, "--message", "hi"],
        ["greencompute", "--base-url", base_url, "invoke", "--model", "wl-1", "--message", "hi", "--stream"],
        ["greencompute", "--base-url", base_url, "workloads", "list"],
    ]

    for argv in commands:
        stdout = io.StringIO()
        old_stdout = sys.stdout
        old_argv = sys.argv
        sys.stdout = stdout
        sys.argv = argv
        try:
            try:
                main()
            except SystemExit as exc:
                assert exc.code == 0
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv
        assert stdout.getvalue().strip()


def test_cli_flags_override_env_and_persisted_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_urls: list[str] = []

    class _CaptureResponse:
        def read(self) -> bytes:
            return json.dumps([{"workload_id": "wl-1", "name": "demo", "image": "demo/echo:latest"}]).encode()

        def __enter__(self) -> "_CaptureResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_capture_urlopen(target, timeout=None):  # type: ignore[no-untyped-def]
        if isinstance(target, str):
            seen_urls.append(target)
        else:
            seen_urls.append(target.full_url)
        return _CaptureResponse()

    config_path = tmp_path / "config.ini"
    config_path.write_text("[greenference]\napi_base_url = http://config.example\napi_key = config-key\n", encoding="utf-8")
    monkeypatch.setenv("GREENCOMPUTE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("GREENCOMPUTE_API_URL", "http://env.example")
    monkeypatch.setenv("GREENCOMPUTE_API_KEY", "env-key")
    monkeypatch.setattr(client_module.request, "urlopen", fake_capture_urlopen)

    old_argv = sys.argv
    sys.argv = ["greencompute", "--base-url", "http://cli.example", "workloads", "list"]
    try:
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = old_argv

    assert seen_urls == ["http://cli.example/platform/workloads"]


def test_cli_wait_commands_exit_nonzero_for_failed_terminal_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_failed_urlopen(target, timeout=None):  # type: ignore[no-untyped-def]
        if isinstance(target, str):
            raise HTTPError(target, 404, "not found", hdrs=None, fp=None)
        path = target.full_url
        method = target.get_method()
        if path.endswith("/platform/builds/build-failed") and method == "GET":
            return _FakeResponse({"build_id": "build-failed", "status": "failed"})
        if path.endswith("/platform/deployments/dep-failed") and method == "GET":
            return _FakeResponse({"deployment_id": "dep-failed", "state": "failed"})
        raise HTTPError(path, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(client_module.request, "urlopen", fake_failed_urlopen)

    for argv in (
        ["greencompute", "builds", "wait", "build-failed"],
        ["greencompute", "deployments", "wait", "dep-failed"],
    ):
        old_argv = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv


def test_cli_prints_http_errors_cleanly_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_http_error(target, timeout=None):  # type: ignore[no-untyped-def]
        raise HTTPError(
            target.full_url,
            403,
            "forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"permission denied"}'),
        )

    monkeypatch.setattr(client_module.request, "urlopen", fake_http_error)

    old_argv = sys.argv
    sys.argv = ["greencompute", "workloads", "get", "wl-1"]
    try:
        with pytest.raises(click.exceptions.Exit):
            main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "HTTP 403" in combined
    assert "permission denied" in combined


def test_client_build_log_entries_capture_terminal_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.request, "urlopen", _fake_urlopen)
    client = client_module.GreenComputeClient(base_url="http://greenference.test")

    entries = list(client.stream_build_log_entries("build-1", follow=False))

    assert entries[-1].terminal is True
    assert entries[-1].status == "published"


def test_cli_share_and_invoke_failures_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_forbidden_urlopen(target, timeout=None):  # type: ignore[no-untyped-def]
        path = target.full_url
        if path.endswith("/platform/workloads/wl-1/shares"):
            raise HTTPError(
                path,
                403,
                "forbidden",
                hdrs=None,
                fp=io.BytesIO(b'{"detail":"share denied"}'),
            )
        if path.endswith("/v1/chat/completions"):
            raise HTTPError(
                path,
                403,
                "forbidden",
                hdrs=None,
                fp=io.BytesIO(b'{"detail":"invoke denied"}'),
            )
        raise HTTPError(path, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(client_module.request, "urlopen", fake_forbidden_urlopen)

    for argv, expected in (
        (["greencompute", "workloads", "share", "wl-1", "--user-id", "user-2"], "share denied"),
        (["greencompute", "invoke", "--model", "wl-1", "--message", "hi"], "invoke denied"),
    ):
        old_argv = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(click.exceptions.Exit):
                main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert expected in combined


def test_cli_deploy_fee_rejection_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fee_urlopen(target, timeout=None):  # type: ignore[no-untyped-def]
        path = target.full_url
        method = target.get_method()
        payload = json.loads(target.data.decode()) if target.data else {}
        if path.endswith("/platform/workloads"):
            return _FakeResponse({"workload_id": "wl-1", "name": payload["name"], "image": payload["image"]})
        if path.endswith("/platform/deployments"):
            raise HTTPError(
                path,
                402,
                "payment required",
                hdrs=None,
                fp=io.BytesIO(b'{"detail":"fee acknowledgement required"}'),
            )
        raise HTTPError(path, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(client_module.request, "urlopen", fake_fee_urlopen)

    old_argv = sys.argv
    sys.argv = [
        "greencompute",
        "deploy",
        "--name",
        "demo",
        "--image",
        "demo/echo:latest",
    ]
    try:
        with pytest.raises(click.exceptions.Exit):
            main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "fee acknowledgement required" in combined
