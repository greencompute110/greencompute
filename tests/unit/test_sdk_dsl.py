from __future__ import annotations

import io
import json
import socket
import warnings
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from greencompute import Image, NodeSelector, RuntimeConfig, Workload
from greencompute.client import (
    BuildInfo,
    BuildLogEntry,
    DeploymentInfo,
    GreenferenceClient,
    GreenferenceHTTPError,
    GreenferenceTimeoutError,
    WarmupEvent,
    WorkloadInfo,
    WorkloadShareInfo,
)
from greencompute.config import get_config, init_config, mask_secret, save_config, unset_config
from greencompute.loader import load_workload
from greencompute.packaging import package_workload
from greencompute.templates import build_vllm_workload
from greencompute.workloads import create_inference_workload


def test_image_workload_loader_and_packaging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "data.txt").write_text("payload", encoding="utf-8")
    (project / "app.py").write_text(
        """
from greencompute import Image, NodeSelector, Workload

image = (
    Image(username="alice", name="demo", tag="latest")
    .from_base("python:3.12-slim")
    .add("data.txt", "/app/data.txt")
    .run_command("echo ready")
)

workload = Workload(
    name="demo-workload",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24, concurrency=4),
    model_identifier="alice/demo-model",
    context_paths=["."],
)
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "ignored.pyc").write_bytes(b"compiled")

    loaded = load_workload(f"{project / 'app.py'}:workload")
    packaged = package_workload(loaded.module_path, loaded.workload)

    assert loaded.workload.image_ref == "alice/demo:latest"
    assert loaded.workload.node_selector.to_requirements_payload()["concurrency"] == 4
    assert loaded.workload.runtime is not None
    assert loaded.workload.runtime.model_identifier == "alice/demo-model"
    assert "RUN echo ready" in packaged.dockerfile_text
    assert packaged.included_paths == ["app.py", "data.txt"]
    assert "__pycache__/ignored.pyc" in packaged.excluded_paths
    assert packaged.archive_size_bytes > 0
    assert packaged.total_input_bytes == len("payload") + len((project / "app.py").read_text(encoding="utf-8").encode())

    with zipfile.ZipFile(io.BytesIO(packaged.archive_bytes)) as archive:
        assert sorted(archive.namelist()) == ["Dockerfile", "app.py", "data.txt"]


def test_config_file_and_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.ini"
    monkeypatch.setenv("GREENFERENCE_CONFIG_PATH", str(config_path))

    saved = save_config(api_base_url="http://saved.example", api_key="saved-key")
    assert saved.api_base_url == "http://saved.example"
    assert saved.api_key == "saved-key"

    monkeypatch.setenv("GREENFERENCE_API_URL", "http://env.example")
    monkeypatch.setenv("GREENFERENCE_API_KEY", "env-key")
    resolved = get_config()

    assert resolved.api_base_url == "http://env.example"
    assert resolved.api_key == "env-key"


def test_config_init_unset_and_masking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.ini"
    monkeypatch.setenv("GREENFERENCE_CONFIG_PATH", str(config_path))

    initialized = init_config(api_base_url="http://init.example", api_key="secret-token")
    assert initialized.api_base_url == "http://init.example"
    assert initialized.api_key == "secret-token"
    assert mask_secret(initialized.api_key) == "secr...en"

    updated = unset_config(api_key=True)
    assert updated.api_base_url == "http://init.example"
    assert updated.api_key is None

    reset = unset_config(api_base_url=True)
    assert reset.api_base_url == "http://127.0.0.1:8000"


def test_template_builder_populates_rich_workload_defaults() -> None:
    workload_pack = build_vllm_workload(
        username="alice",
        name="llm-demo",
        model_identifier="meta-llama/Llama-3.2-1B-Instruct",
        display_name="LLM Demo",
        workload_alias="llm-demo-alias",
        tags=["llm", "chat"],
        context_paths=["README.md"],
    )

    payload = workload_pack.workload.to_workload_payload()

    assert workload_pack.template == "inference"
    assert payload["display_name"] == "LLM Demo"
    assert payload["workload_alias"] == "llm-demo-alias"
    assert payload["runtime"]["runtime_kind"] == "vllm"
    assert payload["lifecycle"]["warmup_enabled"] is True
    assert workload_pack.workload.context_paths == ["README.md"]


def test_runtime_config_and_image_dsl_extensions() -> None:
    image = (
        Image(username="alice", name="extended", tag="latest")
        .with_maintainer("alice@example.com")
        .as_user("appuser")
        .apt_remove(["curl"])
        .copy("README.md", "/app/README.md")
        .with_cmd("python", "-m", "http.server")
    )
    workload = Workload(
        name="runtime-configured",
        image=image,
        runtime=RuntimeConfig(
            runtime_kind="vllm",
            model_identifier="meta-llama/Llama-3.2-1B-Instruct",
            model_revision="main",
        ),
    )

    dockerfile = str(image)
    payload = workload.to_workload_payload()

    assert "LABEL maintainer='alice@example.com'" in dockerfile
    assert "USER appuser" in dockerfile
    assert "RUN apt-get remove -y curl" in dockerfile
    assert "COPY README.md /app/README.md" in dockerfile
    assert 'CMD ["python", "-m", "http.server"]' in dockerfile
    assert payload["runtime"]["runtime_kind"] == "vllm"
    assert payload["runtime"]["model_revision"] == "main"


def test_packaging_rejects_oversized_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "data.txt").write_text("x" * 256, encoding="utf-8")
    (project / "app.py").write_text(
        """
from greencompute import Image, Workload
image = Image(username="alice", name="demo", tag="latest").add("data.txt", "/app/data.txt")
workload = Workload(name="demo-workload", image=image, model_identifier="alice/demo-model")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv("GREENFERENCE_MAX_CONTEXT_ARCHIVE_BYTES", "100")

    loaded = load_workload(f"{project / 'app.py'}:workload")

    with pytest.raises(ValueError, match="exceeds limit"):
        package_workload(loaded.module_path, loaded.workload)


def test_packaging_honors_exclude_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "keep.txt").write_text("keep", encoding="utf-8")
    (project / "ignore.log").write_text("ignore", encoding="utf-8")
    (project / "app.py").write_text(
        """
from greencompute import Image, Workload
image = Image(username="alice", name="demo", tag="latest").add("keep.txt", "/app/keep.txt")
workload = Workload(
    name="demo-workload",
    image=image,
    model_identifier="alice/demo-model",
    context_paths=["."],
    exclude_patterns=["*.log"],
)
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    loaded = load_workload(f"{project / 'app.py'}:workload")
    packaged = package_workload(loaded.module_path, loaded.workload)

    assert "ignore.log" not in packaged.included_paths
    assert "ignore.log" in packaged.excluded_paths


def test_packaging_handles_nested_dirs_and_default_ignores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    nested = project / "pkg" / "data"
    ignored = project / ".pytest_cache"
    nested.mkdir(parents=True)
    ignored.mkdir(parents=True)
    (nested / "keep.json").write_text('{"ok": true}', encoding="utf-8")
    (ignored / "ignored.json").write_text("ignored", encoding="utf-8")
    (project / "app.py").write_text(
        """
from greencompute import Image, Workload
image = Image(username="alice", name="nested", tag="latest").add("pkg/data/keep.json", "/app/keep.json")
workload = Workload(
    name="nested-workload",
    image=image,
    model_identifier="alice/nested-model",
    context_paths=["pkg"],
)
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    loaded = load_workload(f"{project / 'app.py'}:workload")
    packaged = package_workload(loaded.module_path, loaded.workload)

    assert "pkg/data/keep.json" in packaged.included_paths
    assert ".pytest_cache/ignored.json" not in packaged.included_paths
    with zipfile.ZipFile(io.BytesIO(packaged.archive_bytes)) as archive:
        assert "pkg/data/keep.json" in archive.namelist()
        assert ".pytest_cache/ignored.json" not in archive.namelist()


def test_workload_include_paths_extend_context_paths() -> None:
    workload = Workload(
        name="demo",
        image="demo/image:latest",
        model_identifier="demo/model",
        context_paths=["base.txt"],
        include_paths=["extra.txt"],
    )

    assert workload.context_paths == ["base.txt", "extra.txt"]


def test_typed_client_wrappers() -> None:
    client = GreenferenceClient()

    build = client._build_info({"build_id": "b1", "image": "demo:latest", "status": "published"})
    deployment = client._deployment_info(
        {"deployment_id": "d1", "workload_id": "w1", "state": "ready", "requested_instances": 2}
    )
    workload = client._workload_info({"workload_id": "w1", "name": "demo", "image": "demo:latest"})

    assert isinstance(build, BuildInfo)
    assert isinstance(deployment, DeploymentInfo)
    assert isinstance(workload, WorkloadInfo)
    assert build.build_id == "b1"
    assert deployment.requested_instances == 2
    assert workload.name == "demo"


def test_build_log_share_and_warmup_wrappers() -> None:
    client = GreenferenceClient()

    log_entry = client._build_log_entry({"stage": "building", "message": "remote build", "status": "running"})
    share = client._share_info(
        {"share_id": "s1", "workload_id": "w1", "shared_with_user_id": "u2", "permission": "invoke"}
    )
    warmup = client._warmup_event({"workload_id": "w1", "status": "warmup_complete"})

    assert isinstance(log_entry, BuildLogEntry)
    assert isinstance(share, WorkloadShareInfo)
    assert isinstance(warmup, WarmupEvent)
    assert share.permission == "invoke"
    assert warmup.status == "warmup_complete"


def test_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise URLError("temporary network issue")
        return _FakeResponse({"build_id": "b1", "image": "demo:latest", "status": "published"})

    monkeypatch.setattr("greencompute.client.request.urlopen", fake_urlopen)
    client = GreenferenceClient(max_retries=1)

    build = client.get_build("b1")

    assert build["status"] == "published"
    assert attempts["count"] == 2


def test_client_timeout_and_server_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_timeout(req, timeout=None):  # type: ignore[no-untyped-def]
        raise socket.timeout("timed out")

    monkeypatch.setattr("greencompute.client.request.urlopen", fake_timeout)
    client = GreenferenceClient(max_retries=0)

    with pytest.raises(GreenferenceTimeoutError, match="timed out"):
        client.get_build("b1")

    def fake_http_error(req, timeout=None):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, 503, "service unavailable", hdrs=None, fp=io.BytesIO(b'{"detail":"down"}'))

    monkeypatch.setattr("greencompute.client.request.urlopen", fake_http_error)
    with pytest.raises(GreenferenceHTTPError, match="HTTP 503"):
        client.get_build("b1")


def test_compatibility_workloads_module_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        workload_pack = create_inference_workload(
            username="alice",
            name="compat-demo",
            model_identifier="alice/demo-model",
        )

    assert workload_pack.workload.name == "compat-demo"
    assert any(item.category is DeprecationWarning for item in caught)
