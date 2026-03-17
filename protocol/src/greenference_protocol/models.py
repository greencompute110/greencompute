from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from greenference_protocol.enums import DeploymentState, SecurityTier, WorkloadKind


def utcnow() -> datetime:
    return datetime.now(UTC)


class APIKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    user_id: str | None = None
    admin: bool = False
    scopes: list[str] = Field(default_factory=list)


class APIKeyRecord(BaseModel):
    key_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_id: str | None = None
    admin: bool = False
    scopes: list[str] = Field(default_factory=list)
    secret: str
    created_at: datetime = Field(default_factory=utcnow)


class UserRegistrationRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: str | None = None


class UserRecord(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class WorkloadRequirements(BaseModel):
    gpu_count: int = Field(default=1, ge=1, le=8)
    min_vram_gb_per_gpu: int = Field(default=16, ge=1)
    cpu_cores: int = Field(default=8, ge=1)
    memory_gb: int = Field(default=32, ge=1)
    max_instances: int = Field(default=1, ge=1, le=64)
    concurrency: int = Field(default=1, ge=1, le=1024)
    supported_gpu_models: list[str] = Field(default_factory=list)


class WorkloadCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    image: str = Field(min_length=1)
    workload_alias: str | None = Field(default=None, min_length=1, max_length=100)
    ingress_host: str | None = Field(default=None, min_length=1, max_length=255)
    kind: WorkloadKind = WorkloadKind.INFERENCE
    security_tier: SecurityTier = SecurityTier.STANDARD
    pricing_class: str = Field(default="standard", min_length=1, max_length=32)
    requirements: WorkloadRequirements = Field(default_factory=WorkloadRequirements)
    public: bool = False


class WorkloadSpec(WorkloadCreateRequest):
    workload_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utcnow)


class DeploymentCreateRequest(BaseModel):
    workload_id: str
    requested_instances: int = Field(default=1, ge=1, le=64)


class DeploymentRecord(BaseModel):
    deployment_id: str = Field(default_factory=lambda: str(uuid4()))
    workload_id: str
    hotkey: str | None = None
    node_id: str | None = None
    state: DeploymentState = DeploymentState.PENDING
    requested_instances: int = 1
    ready_instances: int = 0
    endpoint: str | None = None
    last_error: str | None = None
    failure_class: str | None = None
    last_retry_reason: str | None = None
    retry_count: int = Field(default=0, ge=0)
    retry_exhausted: bool = False
    health_check_failures: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class NodeCapability(BaseModel):
    hotkey: str
    node_id: str
    server_id: str | None = None
    hostname: str | None = None
    observed_at: datetime | None = None
    gpu_model: str
    gpu_count: int = Field(ge=1, le=8)
    available_gpus: int = Field(ge=0, le=8)
    vram_gb_per_gpu: int = Field(ge=1)
    cpu_cores: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    hourly_cost_usd: float = Field(default=0.0, ge=0.0)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    performance_score: float = Field(default=1.0, ge=0.0)
    security_tier: SecurityTier = SecurityTier.STANDARD
    labels: dict[str, str] = Field(default_factory=dict)


class MinerRegistration(BaseModel):
    hotkey: str
    payout_address: str
    api_base_url: str
    validator_url: str
    auth_secret: str = Field(min_length=8)
    drained: bool = False
    supported_workload_kinds: list[WorkloadKind] = Field(
        default_factory=lambda: [WorkloadKind.INFERENCE]
    )


class Heartbeat(BaseModel):
    hotkey: str
    healthy: bool = True
    active_deployments: int = 0
    active_leases: int = 0
    observed_at: datetime = Field(default_factory=utcnow)


class CapacityUpdate(BaseModel):
    hotkey: str
    nodes: list[NodeCapability]
    observed_at: datetime = Field(default_factory=utcnow)


class ServerRecord(BaseModel):
    server_id: str
    hotkey: str
    hostname: str | None = None
    api_base_url: str | None = None
    validator_url: str | None = None
    observed_at: datetime = Field(default_factory=utcnow)


class CapacityHistoryRecord(BaseModel):
    history_id: str = Field(default_factory=lambda: str(uuid4()))
    hotkey: str
    server_id: str | None = None
    node_id: str
    available_gpus: int = Field(ge=0, le=8)
    total_gpus: int = Field(ge=1, le=8)
    observed_at: datetime = Field(default_factory=utcnow)


class PlacementRecord(BaseModel):
    placement_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    server_id: str | None = None
    node_id: str
    status: str = "assigned"
    reason: str | None = None
    failure_count: int = Field(default=0, ge=0)
    cooldown_until: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class LeaseAssignment(BaseModel):
    assignment_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    node_id: str
    assigned_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    status: str = "assigned"


class LeaseHistoryRecord(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    node_id: str
    status: str
    reason: str | None = None
    observed_at: datetime = Field(default_factory=utcnow)


class DeploymentStatusUpdate(BaseModel):
    deployment_id: str
    state: DeploymentState
    endpoint: str | None = None
    ready_instances: int = Field(default=0, ge=0)
    error: str | None = None
    observed_at: datetime = Field(default_factory=utcnow)


class ProbeChallenge(BaseModel):
    challenge_id: str = Field(default_factory=lambda: str(uuid4()))
    hotkey: str
    node_id: str
    kind: str = "latency"
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ProbeResult(BaseModel):
    challenge_id: str
    hotkey: str
    node_id: str
    latency_ms: float = Field(ge=0.0)
    throughput: float = Field(ge=0.0)
    success: bool = True
    benchmark_signature: str | None = None
    proxy_suspected: bool = False
    readiness_failures: int = Field(default=0, ge=0)
    observed_at: datetime = Field(default_factory=utcnow)


class ScoreCard(BaseModel):
    hotkey: str
    capacity_weight: float = Field(ge=0.0)
    reliability_score: float = Field(ge=0.0)
    performance_score: float = Field(ge=0.0)
    security_score: float = Field(ge=0.0)
    fraud_penalty: float = Field(ge=0.0)
    final_score: float = Field(ge=0.0)
    computed_at: datetime = Field(default_factory=utcnow)


class WeightSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    netuid: int = Field(default=0, ge=0)
    weights: dict[str, float]
    created_at: datetime = Field(default_factory=utcnow)


class UsageRecord(BaseModel):
    deployment_id: str
    workload_id: str
    hotkey: str
    request_count: int = Field(default=1, ge=0)
    streamed_request_count: int = Field(default=0, ge=0)
    stream_chunk_count: int = Field(default=0, ge=0)
    compute_seconds: float = Field(default=0.0, ge=0.0)
    latency_ms_p95: float = Field(default=0.0, ge=0.0)
    occupancy_seconds: float = Field(default=0.0, ge=0.0)
    measured_at: datetime = Field(default_factory=utcnow)


class InvocationRecord(BaseModel):
    invocation_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    model: str
    api_key_id: str | None = None
    stream: bool = False
    status: str = "succeeded"
    error_class: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    message_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class BuildRequest(BaseModel):
    image: str = Field(min_length=1)
    context_uri: str = Field(min_length=1)
    dockerfile_path: str = Field(default="Dockerfile", min_length=1)
    public: bool = False


class BuildRecord(BaseModel):
    build_id: str = Field(default_factory=lambda: str(uuid4()))
    image: str
    context_uri: str
    dockerfile_path: str
    public: bool = False
    status: str = "accepted"
    registry_repository: str | None = None
    image_tag: str | None = None
    artifact_uri: str | None = None
    artifact_digest: str | None = None
    registry_manifest_uri: str | None = None
    build_log_uri: str | None = None
    executor_name: str | None = None
    build_duration_seconds: float | None = None
    failure_reason: str | None = None
    failure_class: str | None = None
    last_operation: str | None = None
    cleanup_status: str | None = None
    retry_count: int = Field(default=0, ge=0)
    retry_exhausted: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class BuildContextRecord(BaseModel):
    build_id: str
    source_uri: str
    normalized_context_uri: str
    dockerfile_path: str
    dockerfile_object_uri: str | None = None
    context_digest: str | None = None
    staged_context_uri: str | None = None
    context_manifest_uri: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class BuildEventRecord(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    build_id: str
    stage: str
    message: str
    created_at: datetime = Field(default_factory=utcnow)


class BuildAttemptRecord(BaseModel):
    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    build_id: str
    attempt: int = Field(ge=1)
    status: str = "accepted"
    failure_class: str | None = None
    last_operation: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class BuildLogRecord(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid4()))
    build_id: str
    attempt: int = Field(ge=1)
    stage: str
    message: str
    created_at: datetime = Field(default_factory=utcnow)


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = Field(default=128, ge=1)
    temperature: float | None = Field(default=0.7, ge=0.0)
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str
    content: str
    deployment_id: str
    routed_hotkey: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
