"""GreenCompute API client."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import quote


class GreenComputeError(Exception):
    """Base exception for GreenCompute client errors."""

    pass


class GreenComputeHTTPError(GreenComputeError):
    """HTTP error (4xx/5xx)."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class GreenComputeConnectionError(GreenComputeError):
    """Connection or network error."""

    pass


class GreenComputeTimeoutError(GreenComputeError):
    """Request timeout."""

    pass


@dataclass(slots=True)
class BuildInfo:
    build_id: str
    image: str
    status: str
    executor_name: str | None = None
    artifact_uri: str | None = None
    artifact_digest: str | None = None
    registry_manifest_uri: str | None = None


@dataclass(slots=True)
class DeploymentInfo:
    deployment_id: str
    workload_id: str
    state: str
    requested_instances: int = 0
    endpoint: str | None = None
    fee_acknowledged: bool | None = None


@dataclass(slots=True)
class WorkloadInfo:
    workload_id: str
    name: str
    image: str
    display_name: str | None = None
    public: bool = False
    workload_alias: str | None = None


@dataclass(slots=True)
class BuildLogEntry:
    stage: str | None = None
    message: str | None = None
    status: str | None = None
    build_id: str | None = None
    event: str | None = None
    terminal: bool = False


@dataclass(slots=True)
class WorkloadShareInfo:
    share_id: str
    workload_id: str
    shared_with_user_id: str
    permission: str


@dataclass(slots=True)
class WarmupEvent:
    workload_id: str
    status: str
    detail: dict | None = None


class GreenComputeClient:
    """HTTP client for the GreenCompute API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"content-type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
            h["X-API-Key"] = self.api_key
        return h

    def _open(self, req: request.Request) -> object:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return request.urlopen(req, timeout=self.timeout_seconds)
            except (TimeoutError, socket.timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise GreenComputeTimeoutError(str(exc)) from exc
            except HTTPError as exc:
                last_exc = exc
                if 500 <= exc.code < 600 and attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                body = exc.fp.read().decode() if exc.fp else None
                raise GreenComputeHTTPError(
                    f"HTTP {exc.code}: {exc.reason}",
                    status_code=exc.code,
                    body=body,
                ) from exc
            except URLError as exc:
                last_exc = exc
                if isinstance(exc.reason, (TimeoutError, OSError, socket.timeout)):
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    raise GreenComputeTimeoutError(str(exc)) from exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise GreenComputeConnectionError(str(exc)) from exc
        if last_exc:
            raise GreenComputeError(str(last_exc)) from last_exc
        raise GreenComputeError("request failed")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict | list:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = request.Request(
            url=url,
            data=data,
            headers=self._headers(),
            method=method,
        )
        with self._open(req) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}

    def _get(self, path: str) -> dict | list:
        req = request.Request(
            url=f"{self.base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        with self._open(req) as response:
            return json.loads(response.read().decode())

    def _post(self, path: str, payload: dict) -> dict | list:
        return self._request("POST", path, payload)

    def _patch(self, path: str, payload: dict) -> dict | list:
        return self._request("PATCH", path, payload)

    def _delete(self, path: str) -> dict | list:
        req = request.Request(
            url=f"{self.base_url}{path}",
            headers=self._headers(),
            method="DELETE",
        )
        with self._open(req) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}

    def _post_stream(self, path: str, payload: dict) -> Iterator[str]:
        raw = json.dumps(payload).encode()
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=raw,
            headers=self._headers(),
            method="POST",
        )
        with self._open(req) as response:
            for line in response.read().decode().splitlines():
                if not line.startswith("data: "):
                    continue
                yield line[6:]

    def _get_stream(self, path: str) -> Iterator[tuple[str | None, str]]:
        req = request.Request(
            url=f"{self.base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        with self._open(req) as response:
            current_event: str | None = None
            for raw_line in response.read().decode().splitlines():
                line = raw_line.strip()
                if not line:
                    current_event = None
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if line.startswith("data: "):
                    yield current_event, line[6:].strip()

    @staticmethod
    def _build_info(data: dict) -> BuildInfo:
        return BuildInfo(
            build_id=str(data.get("build_id", "")),
            image=str(data.get("image", "")),
            status=str(data.get("status", "")),
            executor_name=data.get("executor_name"),
            artifact_uri=data.get("artifact_uri"),
            artifact_digest=data.get("artifact_digest"),
            registry_manifest_uri=data.get("registry_manifest_uri"),
        )

    @staticmethod
    def _deployment_info(data: dict) -> DeploymentInfo:
        return DeploymentInfo(
            deployment_id=str(data.get("deployment_id", "")),
            workload_id=str(data.get("workload_id", "")),
            state=str(data.get("state", "")),
            requested_instances=int(data.get("requested_instances", 0) or 0),
            endpoint=data.get("endpoint"),
            fee_acknowledged=data.get("fee_acknowledged"),
        )

    @staticmethod
    def _workload_info(data: dict) -> WorkloadInfo:
        return WorkloadInfo(
            workload_id=str(data.get("workload_id", "")),
            name=str(data.get("name", "")),
            image=str(data.get("image", "")),
            display_name=data.get("display_name"),
            public=bool(data.get("public", False)),
            workload_alias=data.get("workload_alias"),
        )

    @staticmethod
    def _build_log_entry(data: dict) -> BuildLogEntry:
        return BuildLogEntry(
            stage=data.get("stage"),
            message=data.get("message"),
            status=data.get("status"),
            build_id=data.get("build_id"),
        )

    @staticmethod
    def _share_info(data: dict) -> WorkloadShareInfo:
        return WorkloadShareInfo(
            share_id=str(data.get("share_id", "")),
            workload_id=str(data.get("workload_id", "")),
            shared_with_user_id=str(data.get("shared_with_user_id", "")),
            permission=str(data.get("permission", "")),
        )

    @staticmethod
    def _warmup_event(data: dict) -> WarmupEvent:
        return WarmupEvent(
            workload_id=str(data.get("workload_id", "")),
            status=str(data.get("status", "")),
            detail=data,
        )

    # --- Auth-free ---
    def register(self, payload: dict) -> dict:
        return self._post("/platform/register", payload)  # type: ignore[return-value]

    # --- API Keys ---
    def create_api_key(self, payload: dict) -> dict:
        return self._post("/platform/api-keys", payload)  # type: ignore[return-value]

    def list_api_keys(self) -> list[dict]:
        return self._get("/platform/api-keys")  # type: ignore[return-value]

    def get_api_key(self, key_id: str) -> dict:
        return self._get(f"/platform/api-keys/{key_id}")  # type: ignore[return-value]

    def delete_api_key(self, key_id: str) -> dict:
        return self._delete(f"/platform/api-keys/{key_id}")  # type: ignore[return-value]

    # --- Users ---
    def get_user(self, user_id: str) -> dict:
        return self._get(f"/platform/users/{user_id}")  # type: ignore[return-value]

    def get_user_balance(self, user_id: str) -> dict:
        return self._get(f"/platform/users/{user_id}/balance")  # type: ignore[return-value]

    # --- Images / Builds ---
    def upload_build_context(self, payload: dict) -> dict:
        return self._post("/platform/images/contexts", payload)  # type: ignore[return-value]

    def build(self, payload: dict) -> dict:
        return self._post("/platform/images", payload)  # type: ignore[return-value]

    def list_images(self) -> list[dict]:
        return self._get("/platform/images")  # type: ignore[return-value]

    def list_image_history(self, image: str) -> list[dict]:
        return self._get(f"/platform/images/{quote(image, safe='')}/history")  # type: ignore[return-value]

    def list_builds(self) -> list[dict]:
        return self._get("/platform/builds")  # type: ignore[return-value]

    def list_build_infos(self) -> list[BuildInfo]:
        return [self._build_info(item) for item in self.list_builds()]

    def get_build(self, build_id: str) -> dict:
        return self._get(f"/platform/builds/{build_id}")  # type: ignore[return-value]

    def get_build_info(self, build_id: str) -> BuildInfo:
        return self._build_info(self.get_build(build_id))

    def stream_build_logs(self, build_id: str, *, follow: bool = False) -> Iterator[dict]:
        for _, payload in self._get_stream(
            f"/platform/builds/{build_id}/logs/stream?follow={'true' if follow else 'false'}"
        ):
            if payload:
                yield json.loads(payload)

    def stream_build_log_entries(self, build_id: str, *, follow: bool = False) -> Iterator[BuildLogEntry]:
        for event_name, payload in self._get_stream(
            f"/platform/builds/{build_id}/logs/stream?follow={'true' if follow else 'false'}"
        ):
            if not payload:
                continue
            entry = self._build_log_entry(json.loads(payload))
            entry.event = event_name
            if event_name == "end" or entry.status in {"published", "failed", "cancelled"}:
                entry.terminal = True
            yield entry

    def wait_for_build(
        self,
        build_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict:
        started = time.monotonic()
        while True:
            build = self.get_build(build_id)
            status = build.get("status")
            if status in {"published", "failed", "cancelled"}:
                return build
            if time.monotonic() - started > timeout_seconds:
                raise GreenComputeTimeoutError(f"timed out waiting for build {build_id}")
            time.sleep(poll_interval_seconds)

    def wait_for_build_info(
        self,
        build_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> BuildInfo:
        return self._build_info(
            self.wait_for_build(
                build_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        )

    def wait_for_deployment(
        self,
        deployment_id: str,
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 2.0,
    ) -> dict:
        started = time.monotonic()
        while True:
            deployment = self.get_deployment(deployment_id)
            state = str(deployment.get("state", "")).lower()
            if state in {"ready", "failed", "terminated"}:
                return deployment
            if time.monotonic() - started > timeout_seconds:
                raise GreenComputeTimeoutError(f"timed out waiting for deployment {deployment_id}")
            time.sleep(poll_interval_seconds)

    def wait_for_deployment_info(
        self,
        deployment_id: str,
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 2.0,
    ) -> DeploymentInfo:
        return self._deployment_info(
            self.wait_for_deployment(
                deployment_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        )

    def wait_for_workload_ready(
        self,
        workload_id: str,
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 2.0,
    ) -> DeploymentInfo:
        started = time.monotonic()
        while True:
            for deployment in self.list_deployment_infos():
                if deployment.workload_id != workload_id:
                    continue
                if deployment.state.lower() in {"ready", "failed", "terminated"}:
                    return deployment
            if time.monotonic() - started > timeout_seconds:
                raise GreenComputeTimeoutError(f"timed out waiting for workload {workload_id} to become ready")
            time.sleep(poll_interval_seconds)

    # --- Workloads ---
    def create_workload(self, payload: dict) -> dict:
        return self._post("/platform/workloads", payload)  # type: ignore[return-value]

    def list_workloads(self) -> list[dict]:
        return self._get("/platform/workloads")  # type: ignore[return-value]

    def list_workload_infos(self) -> list[WorkloadInfo]:
        return [self._workload_info(item) for item in self.list_workloads()]

    def get_workload(self, workload_id: str) -> dict:
        return self._get(f"/platform/workloads/{workload_id}")  # type: ignore[return-value]

    def get_workload_info(self, workload_id: str) -> WorkloadInfo:
        return self._workload_info(self.get_workload(workload_id))

    def update_workload(self, workload_id: str, payload: dict) -> dict:
        return self._patch(f"/platform/workloads/{workload_id}", payload)  # type: ignore[return-value]

    def delete_workload(self, workload_id: str) -> dict:
        return self._delete(f"/platform/workloads/{workload_id}")  # type: ignore[return-value]

    def get_workload_utilization(self, workload_id: str) -> dict:
        return self._get(f"/platform/workloads/{workload_id}/utilization")  # type: ignore[return-value]

    def share_workload(self, workload_id: str, payload: dict) -> dict:
        return self._post(f"/platform/workloads/{workload_id}/shares", payload)  # type: ignore[return-value]

    def list_workload_shares(self, workload_id: str) -> list[dict]:
        return self._get(f"/platform/workloads/{workload_id}/shares")  # type: ignore[return-value]

    def list_workload_share_infos(self, workload_id: str) -> list[WorkloadShareInfo]:
        return [self._share_info(item) for item in self.list_workload_shares(workload_id)]

    def workload_warmup(self, workload_id: str) -> Iterator[dict]:
        """Stream warmup events (SSE) for a workload."""
        for _, payload in self._get_stream(f"/platform/workloads/{workload_id}/warmup"):
            if payload:
                yield json.loads(payload)

    def workload_warmup_events(self, workload_id: str) -> Iterator[WarmupEvent]:
        for event in self.workload_warmup(workload_id):
            yield self._warmup_event(event)

    def guess_vllm_config(self, model: str) -> dict:
        """Analyze HuggingFace model and return GPU requirements (VRAM, GPU count, etc.)."""
        return self._get(f"/guess/vllm_config?model={quote(model, safe='')}")  # type: ignore[return-value]

    # --- Deployments ---
    def deploy(self, payload: dict) -> dict:
        return self._post("/platform/deployments", payload)  # type: ignore[return-value]

    def list_deployments(self) -> list[dict]:
        return self._get("/platform/deployments")  # type: ignore[return-value]

    def list_deployment_infos(self) -> list[DeploymentInfo]:
        return [self._deployment_info(item) for item in self.list_deployments()]

    def get_deployment(self, deployment_id: str) -> dict:
        return self._get(f"/platform/deployments/{deployment_id}")  # type: ignore[return-value]

    def get_deployment_info(self, deployment_id: str) -> DeploymentInfo:
        return self._deployment_info(self.get_deployment(deployment_id))

    def update_deployment(self, deployment_id: str, payload: dict) -> dict:
        return self._patch(f"/platform/deployments/{deployment_id}", payload)  # type: ignore[return-value]

    # --- Secrets ---
    def create_secret(self, payload: dict) -> dict:
        return self._post("/platform/secrets", payload)  # type: ignore[return-value]

    def list_secrets(self) -> list[dict]:
        return self._get("/platform/secrets")  # type: ignore[return-value]

    def delete_secret(self, secret_id: str) -> dict:
        return self._delete(f"/platform/secrets/{secret_id}")  # type: ignore[return-value]

    # --- Inference ---
    def invoke(self, payload: dict) -> dict:
        return self._post("/v1/chat/completions", payload)  # type: ignore[return-value]

    def invoke_stream(self, payload: dict) -> Iterator[str]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        return self._post_stream("/v1/chat/completions", stream_payload)

    def invoke_workload(
        self,
        workload: str,
        *,
        message: str,
        stream: bool = False,
        system_message: str | None = None,
    ) -> dict | Iterator[str]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})
        payload = {"model": workload, "messages": messages}
        if stream:
            return self.invoke_stream(payload)
        return self.invoke(payload)

    def completions(self, payload: dict) -> dict:
        return self._post("/v1/completions", payload)  # type: ignore[return-value]

    def embeddings(self, payload: dict) -> dict:
        return self._post("/v1/embeddings", payload)  # type: ignore[return-value]

    # --- Aliases ---
    def workloads(self) -> list[dict]:
        return self.list_workloads()

    def register_miner(self, payload: dict) -> dict:
        return self._post("/agent/v1/register", payload)  # type: ignore[return-value]
