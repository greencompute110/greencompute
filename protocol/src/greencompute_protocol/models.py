from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from greencompute_protocol.enums import (
    DeploymentState,
    FluxDecision,
    GpuAllocationMode,
    SecurityTier,
    WorkloadKind,
)


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
    secret: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class APIKeySummary(BaseModel):
    """API key without secret (for list/get responses)."""

    key_id: str
    name: str
    user_id: str | None = None
    admin: bool = False
    scopes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class UserRegistrationRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: str | None = None


class UserProfileUpdateRequest(BaseModel):
    email: str | None = None
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    bio: str | None = Field(default=None, max_length=1024)
    website: str | None = Field(default=None, min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserRecord(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str | None = None
    display_name: str | None = None
    bio: str | None = None
    website: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    balance_credits: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class WorkloadRequirements(BaseModel):
    gpu_count: int = Field(default=1, ge=1, le=8)
    min_vram_gb_per_gpu: int = Field(default=16, ge=1)
    cpu_cores: int = Field(default=8, ge=1)
    memory_gb: int = Field(default=32, ge=1)
    max_instances: int = Field(default=1, ge=1, le=64)
    concurrency: int = Field(default=1, ge=1, le=1024)
    supported_gpu_models: list[str] = Field(default_factory=list)


class InferenceRuntimeConfig(BaseModel):
    runtime_kind: str = Field(default="hf-causal-lm", min_length=1, max_length=64)
    model_identifier: str = Field(default="sshleifer/tiny-gpt2", min_length=1, max_length=255)
    model_revision: str | None = Field(default=None, min_length=1, max_length=128)
    tokenizer_identifier: str | None = Field(default=None, min_length=1, max_length=255)


class WorkloadLifecyclePolicy(BaseModel):
    scaling_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    shutdown_after_seconds: int = Field(default=300, ge=0, le=86400)
    warmup_enabled: bool = False
    warmup_path: str | None = Field(default=None, min_length=1, max_length=255)


class WorkloadCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    image: str = Field(min_length=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    readme: str | None = Field(default=None, max_length=20000)
    logo_uri: str | None = Field(default=None, min_length=1, max_length=1024)
    tags: list[str] = Field(default_factory=list)
    workload_alias: str | None = Field(default=None, min_length=1, max_length=100)
    ingress_host: str | None = Field(default=None, min_length=1, max_length=255)
    kind: WorkloadKind = WorkloadKind.INFERENCE
    security_tier: SecurityTier = SecurityTier.STANDARD
    pricing_class: str = Field(default="standard", min_length=1, max_length=32)
    requirements: WorkloadRequirements = Field(default_factory=WorkloadRequirements)
    runtime: InferenceRuntimeConfig = Field(default_factory=InferenceRuntimeConfig)
    lifecycle: WorkloadLifecyclePolicy = Field(default_factory=WorkloadLifecyclePolicy)
    public: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkloadUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    readme: str | None = Field(default=None, max_length=20000)
    logo_uri: str | None = Field(default=None, min_length=1, max_length=1024)
    tags: list[str] | None = None
    workload_alias: str | None = Field(default=None, min_length=1, max_length=100)
    clear_workload_alias: bool = False
    ingress_host: str | None = Field(default=None, min_length=1, max_length=255)
    pricing_class: str | None = Field(default=None, min_length=1, max_length=32)
    public: bool | None = None
    lifecycle: WorkloadLifecyclePolicy | None = None


class WorkloadSpec(WorkloadCreateRequest):
    workload_id: str = Field(default_factory=lambda: str(uuid4()))
    owner_user_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class DeploymentCreateRequest(BaseModel):
    workload_id: str
    requested_instances: int = Field(default=1, ge=1, le=64)
    accept_fee: bool = True


class DeploymentUpdateRequest(BaseModel):
    requested_instances: int | None = Field(default=None, ge=1, le=64)
    fee_acknowledged: bool | None = None


class DeploymentRecord(BaseModel):
    deployment_id: str = Field(default_factory=lambda: str(uuid4()))
    workload_id: str
    owner_user_id: str | None = None
    hotkey: str | None = None
    node_id: str | None = None
    state: DeploymentState = DeploymentState.PENDING
    requested_instances: int = 1
    ready_instances: int = 0
    endpoint: str | None = None
    ssh_private_key: str | None = None
    port_mappings: dict[int, int] = Field(default_factory=dict)
    # Cents per GPU per hour — locked at placement from the node's gpu_model.
    # Default 10 preserves the legacy $0.10/hr behaviour for any row that
    # somehow skips the placement hook.
    hourly_rate_cents: int = Field(default=10, ge=0)
    deployment_fee_usd: float = Field(default=0.0, ge=0.0)
    fee_acknowledged: bool = True
    warmup_state: str = "pending"
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
    ssh_private_key: str | None = None
    port_mappings: dict[int, int] | None = None
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
    # A.5 probe hardening: prompt/response digests so auditors can verify
    # the miner actually served a nonce-bearing prompt instead of a cached
    # canned response. Nullable for back-compat with old probe results.
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    observed_at: datetime = Field(default_factory=utcnow)


class ScoreCard(BaseModel):
    hotkey: str
    capacity_weight: float = Field(ge=0.0)
    reliability_score: float = Field(ge=0.0)
    performance_score: float = Field(ge=0.0)
    security_score: float = Field(ge=0.0)
    fraud_penalty: float = Field(ge=0.0)
    utilization_score: float = Field(default=1.0, ge=0.0)
    rental_revenue_bonus: float = Field(default=0.0, ge=0.0)
    final_score: float = Field(ge=0.0)
    computed_at: datetime = Field(default_factory=utcnow)


class WeightSnapshot(BaseModel):
    """A single weight-vector computed by the validator, tied to a netuid.

    GreenCompute's netuids: 110 on mainnet (finney), 16 on testnet."""

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
    routed_host: str | None = None
    resolution_basis: str | None = None
    routing_reason: str | None = None
    stream: bool = False
    status: str = "succeeded"
    error_class: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    message_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class UserSecretCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=4096)


class UserSecretRecord(BaseModel):
    secret_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    value: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class WorkloadShareCreateRequest(BaseModel):
    shared_with_user_id: str = Field(min_length=1, max_length=64)
    permission: str = Field(default="invoke", min_length=1, max_length=32)


class WorkloadShareRecord(BaseModel):
    share_id: str = Field(default_factory=lambda: str(uuid4()))
    workload_id: str
    owner_user_id: str
    shared_with_user_id: str
    permission: str = "invoke"
    created_at: datetime = Field(default_factory=utcnow)


class BuildRequest(BaseModel):
    image: str = Field(min_length=1)
    context_uri: str | None = None
    dockerfile_path: str = Field(default="Dockerfile", min_length=1)
    context_archive_b64: str | None = None
    context_archive_name: str | None = Field(default=None, min_length=1, max_length=255)
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    readme: str | None = Field(default=None, max_length=20000)
    logo_uri: str | None = Field(default=None, min_length=1, max_length=1024)
    tags: list[str] = Field(default_factory=list)
    public: bool = False


class BuildContextUploadRequest(BaseModel):
    context_archive_b64: str = Field(min_length=1)
    context_archive_name: str = Field(min_length=1, max_length=255)


class BuildContextUploadRecord(BaseModel):
    context_uri: str
    archive_name: str
    size_bytes: int = Field(ge=0)
    uploaded_at: datetime = Field(default_factory=utcnow)


class BuildRecord(BaseModel):
    build_id: str = Field(default_factory=lambda: str(uuid4()))
    image: str
    owner_user_id: str | None = None
    context_uri: str
    dockerfile_path: str
    display_name: str | None = None
    readme: str | None = None
    logo_uri: str | None = None
    tags: list[str] = Field(default_factory=list)
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
    restarted_from_attempt: int | None = None
    restarted_from_job_id: str | None = None
    restart_reason: str | None = None
    failure_class: str | None = None
    last_operation: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class BuildJobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    build_id: str
    attempt: int = Field(ge=1)
    status: str = "queued"
    current_stage: str = "accepted"
    last_completed_stage: str | None = None
    stage_state: dict[str, Any] = Field(default_factory=dict)
    restarted_from_attempt: int | None = None
    restarted_from_job_id: str | None = None
    restart_reason: str | None = None
    executor_name: str | None = None
    failure_class: str | None = None
    progress_message: str | None = None
    recovery_count: int = Field(default=0, ge=0)
    last_recovered_at: datetime | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class BuildJobCheckpointRecord(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    build_id: str
    attempt: int = Field(ge=1)
    stage: str
    status: str
    message: str
    recovered: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class BuildLogRecord(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid4()))
    build_id: str
    attempt: int = Field(ge=1)
    stage: str
    message: str
    created_at: datetime = Field(default_factory=utcnow)


class PodConfig(BaseModel):
    """Pod-specific config — Lium-style: template, SSH keys, volumes, TTL, GPU splitting."""

    template: str | None = Field(default=None, max_length=64)
    ssh_public_keys: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)
    volume_size_gb: int = Field(default=50, ge=1, le=2048)
    gpu_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    capacity_type: str = Field(default="reserved")
    shutdown_after_seconds: int = Field(default=0, ge=0)


class SSHAccessRecord(BaseModel):
    deployment_id: str
    host: str
    port: int
    username: str = "user"
    private_key: str | None = None
    fingerprint: str | None = None
    ready: bool = False


class ComputeRuntimeRecord(BaseModel):
    runtime_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    node_id: str
    workload_kind: str
    status: str = "accepted"
    current_stage: str = "accepted_lease"
    endpoint: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str = "user"
    ssh_fingerprint: str | None = None
    # Extra user-exposed ports: {container_port: host_port}. Populated by the
    # node-agent after `docker run -p` completes. Max 10 ports (enforced pod-side).
    port_mappings: dict[int, int] = Field(default_factory=dict)
    # Per-pod resource enforcement (Docker --cpus / --memory). Computed as
    # host_total * (gpu_count / host_gpu_count). 0 means unbounded (legacy).
    cpu_cores_allocated: float = 0.0
    memory_gb_allocated: int = 0
    volume_id: str | None = None
    volume_path: str | None = None
    volume_size_gb: int = 50
    gpu_fraction: float = 1.0
    container_id: str | None = None
    vm_id: str | None = None
    template: str | None = None
    ttl_seconds: int = 0
    failure_class: str | None = None
    last_error: str | None = None
    restart_count: int = Field(default=0, ge=0)
    last_healthcheck_at: datetime | None = None
    last_transition_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VolumeRecord(BaseModel):
    volume_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    hotkey: str
    node_id: str
    path: str
    size_gb: int
    backup_uri: str | None = None
    last_backed_up_at: datetime | None = None
    status: str = "created"
    created_at: datetime = Field(default_factory=utcnow)


class CollateralRecord(BaseModel):
    hotkey: str
    amount_tao: float = Field(default=0.0, ge=0.0)
    locked: bool = False
    slash_events: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


class ComputePlacementRecord(BaseModel):
    placement_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    runtime_id: str | None = None
    hotkey: str
    node_id: str
    server_id: str | None = None
    hostname: str | None = None
    status: str = "assigned"
    reason: str | None = None
    assigned_at: datetime = Field(default_factory=utcnow)
    activated_at: datetime | None = None
    released_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class ChatCompletionContentBlock(BaseModel):
    """OpenAI-compatible content block for multimodal messages.

    Supported `type` values:
      - "text"       → uses `text`
      - "image_url"  → uses `image_url` = {"url": "https://..." or "data:image/...;base64,..."}
    Additional keys are preserved for forward compatibility with vLLM's multimodal spec
    (video_url, input_audio, etc.) via `model_config`.
    """

    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ChatCompletionMessage(BaseModel):
    role: str
    # OpenAI spec: content is either a plain string OR a list of content blocks
    # (for multimodal — images, audio, video). Qwen2-VL, LLaVA, etc. require this form.
    content: str | list[ChatCompletionContentBlock]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = Field(default=2048, ge=1)
    temperature: float | None = Field(default=0.7, ge=0.0)
    stream: bool = False
    # OpenAI-compatible: {"include_usage": true} on streaming requests asks
    # vLLM to emit a final chunk with `usage.{prompt,completion,total}_tokens`
    # so we can charge after the stream ends. The gateway injects this
    # automatically when stream=True; users can override if they need to.
    stream_options: dict | None = None
    # Pass-through for additional OpenAI fields that vLLM supports
    # (e.g. top_p, frequency_penalty, stop, user, etc.).
    model_config = {"extra": "allow"}


class ChatCompletionUsage(BaseModel):
    """OpenAI-style token usage breakdown returned by the miner.

    prompt_tokens + completion_tokens = total_tokens. The gateway multiplies
    prompt_tokens by INFERENCE_INPUT_CENTS_PER_MTOK and completion_tokens by
    INFERENCE_OUTPUT_CENTS_PER_MTOK to compute the per-request charge.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """OpenAI-format choice. vLLM returns these fields natively."""

    index: int = 0
    message: ChatCompletionMessage | None = None
    finish_reason: str | None = None

    # vLLM-specific fields we accept but don't require (reasoning_content,
    # tool_calls, logprobs, stop_reason). `extra=allow` passes them through
    # so OpenAI-compatible clients see the same payload the miner sent.
    model_config = {"extra": "allow"}


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response. Intentionally permissive
    so any OpenAI-compatible server (vLLM, TGI, llama.cpp) can be the upstream
    without translation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage | None = None

    # Gateway-injected diagnostics.
    deployment_id: str | None = None
    routed_hotkey: str | None = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Flux orchestrator models
# ---------------------------------------------------------------------------


class FluxState(BaseModel):
    """Per-miner GPU allocation state tracked by the Flux orchestrator."""

    hotkey: str
    node_id: str
    total_gpus: int = Field(ge=0)
    inference_gpus: int = Field(default=0, ge=0)
    rental_gpus: int = Field(default=0, ge=0)
    idle_gpus: int = Field(default=0, ge=0)
    inference_floor_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    rental_floor_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    inference_demand_score: float = Field(default=0.0, ge=0.0)
    rental_demand_score: float = Field(default=0.0, ge=0.0)
    # Per-miner catalog-model assignments: model_id → list of GPU indices
    # allocated to that model. e.g. {"qwen-7b": [0,1,2,3]} means GPUs 0..3 on
    # this miner should host qwen-7b. Miners consume this in their reconcile
    # loop to spawn/kill containers autonomously (Phase 2D).
    inference_assignments: dict[str, list[int]] = Field(default_factory=dict)
    last_rebalanced_at: datetime | None = None


class FluxRebalanceEvent(BaseModel):
    """Audit record of a GPU mode transition decided by Flux."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    hotkey: str
    node_id: str
    gpu_index: int = Field(ge=0)
    from_mode: GpuAllocationMode
    to_mode: GpuAllocationMode
    reason: str
    # Optional: which catalog model this GPU was assigned to / released from.
    # Set when transitioning into or out of INFERENCE mode with a known
    # catalog target. None for RENTAL / IDLE transitions.
    model_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class RentalWaitEstimate(BaseModel):
    """Returned to users when a rental GPU is busy with inference."""

    deployment_id: str
    estimated_wait_seconds: float = Field(ge=0.0)
    gpu_currently_serving: str | None = None
    position_in_queue: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Unified runtime record (superset of inference + compute fields)
# ---------------------------------------------------------------------------


class UnifiedRuntimeRecord(BaseModel):
    """Single runtime record for any workload kind (inference, pod, VM)."""

    runtime_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    workload_id: str
    hotkey: str
    node_id: str
    workload_kind: WorkloadKind
    status: str = "accepted"
    current_stage: str = "accepted"
    endpoint: str | None = None

    # Inference-specific fields
    build_id: str | None = None
    image: str | None = None
    artifact_uri: str | None = None
    artifact_digest: str | None = None
    staged_artifact_path: str | None = None
    runtime_dir: str | None = None
    runtime_url: str | None = None
    process_id: int | None = None
    runtime_mode: str | None = None
    backend_name: str | None = None
    model_identifier: str | None = None

    # Compute-specific fields (pod/VM)
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str = "user"
    ssh_fingerprint: str | None = None
    # User-exposed TCP ports: {container_port: host_port}. Set by node-agent
    # after the pod is running. Max 10 ports.
    port_mappings: dict[int, int] = Field(default_factory=dict)
    # Enforced resource limits on the pod (Docker --cpus / --memory).
    cpu_cores_allocated: float = 0.0
    memory_gb_allocated: int = 0
    volume_id: str | None = None
    volume_path: str | None = None
    volume_size_gb: int = 50
    gpu_fraction: float = 1.0
    container_id: str | None = None
    vm_id: str | None = None
    template: str | None = None
    ttl_seconds: int = 0

    # Shared fields
    server_id: str | None = None
    failure_class: str | None = None
    last_error: str | None = None
    restart_count: int = Field(default=0, ge=0)
    crash_count: int = Field(default=0, ge=0)
    recovery_count: int = Field(default=0, ge=0)
    last_healthcheck_at: datetime | None = None
    last_transition_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bittensor chain integration models
# ---------------------------------------------------------------------------


class MetagraphEntry(BaseModel):
    """A single neuron from the Bittensor metagraph."""

    uid: int = Field(ge=0)
    hotkey: str
    coldkey: str
    stake: float = Field(default=0.0, ge=0.0)
    incentive: float = Field(default=0.0, ge=0.0)
    emission: float = Field(default=0.0, ge=0.0)
    registered: bool = True


class MinerWhitelistEntry(BaseModel):
    """Approved miner — only whitelisted hotkeys receive incentive."""

    hotkey: str
    label: str = ""
    energy_source: str = ""
    notes: str = ""
    approved_at: datetime = Field(default_factory=utcnow)


class GreenEnergyApplication(BaseModel):
    """Provider application to join the subnet with green-energy proof."""

    application_id: str = Field(default_factory=lambda: str(uuid4()))
    hotkey: str
    signature: str = ""
    organization: str = ""
    energy_source: str = ""
    description: str = ""
    status: str = "pending"  # pending | approved | rejected
    reviewer_notes: str = ""
    submitted_at: datetime = Field(default_factory=utcnow)
    reviewed_at: datetime | None = None


class GreenEnergyAttachment(BaseModel):
    """File attached to a green-energy application."""

    attachment_id: str = Field(default_factory=lambda: str(uuid4()))
    application_id: str
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    data_b64: str = ""
    uploaded_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Model catalog — Chutes-style shared inference pool
# ---------------------------------------------------------------------------


class ModelCatalogEntry(BaseModel):
    """An admin-approved model that the subnet's inference pool hosts.

    Each catalog entry corresponds to exactly one canonical WorkloadSpec
    (lookup key: workload.name == model_id). Multiple miners host the same
    workload — one DeploymentRecord per serving miner. Callers reach any
    replica via POST /v1/chat/completions with model=<model_id>.
    """

    # Pydantic v2 reserves the `model_*` namespace for internals; opt out so
    # `model_id`, `model_len` etc. aren't treated as shadowing.
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(min_length=1, max_length=128)
    display_name: str = ""
    hf_repo: str = ""
    template: str = "vllm"  # "vllm" | "vllm-vision" | "diffusion"
    min_vram_gb_per_gpu: int = Field(default=24, ge=1)
    gpu_count: int = Field(default=1, ge=1, le=8)
    max_model_len: int | None = Field(default=None, ge=1)
    visibility: str = "public"  # "public" | "gated"
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int | None = Field(default=None, ge=1)
    admin_notes: str = ""
    created_by_hotkey: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class CatalogSubmission(BaseModel):
    """Miner-proposed addition to the model catalog. Admin approves/rejects;
    approval auto-creates a ModelCatalogEntry + canonical workload."""

    model_config = {"protected_namespaces": ()}

    submission_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str = Field(min_length=1, max_length=128)
    hotkey: str = ""
    signature: str = ""
    display_name: str = ""
    hf_repo: str = ""
    template: str = "vllm"
    min_vram_gb_per_gpu: int = Field(default=24, ge=1)
    gpu_count: int = Field(default=1, ge=1, le=8)
    max_model_len: int | None = Field(default=None, ge=1)
    rationale: str = ""
    status: str = "pending"  # pending | approved | rejected
    reviewer_notes: str = ""
    submitted_at: datetime = Field(default_factory=utcnow)
    reviewed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Billing models
# ---------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """Immutable audit record of a balance change."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    amount_cents: int
    balance_after: int
    kind: str  # topup | usage | refund | bonus
    reference_id: str | None = None
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class CryptoInvoice(BaseModel):
    """Crypto deposit invoice."""

    invoice_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    currency: str  # usdt | usdc | tao | alpha
    amount_crypto: float
    amount_usd: float
    bonus_pct: float = 0.0
    total_credits: int = 0
    deposit_address: str = ""
    status: str = "pending"  # pending | confirmed | expired | cancelled
    tx_hash: str | None = None
    expires_at: datetime = Field(default_factory=utcnow)
    confirmed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)


class StripeSession(BaseModel):
    """Stripe checkout session record."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    stripe_session_id: str = ""
    amount_usd: float = 0.0
    amount_cents: int = 0
    status: str = "pending"  # pending | paid | expired
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None


class ChainWeightCommit(BaseModel):
    """Record of a set_weights extrinsic submitted to the chain."""

    commit_id: str = Field(default_factory=lambda: str(uuid4()))
    netuid: int = Field(ge=0)
    uids: list[int]
    weights: list[float]
    version_key: int = 0
    tx_hash: str | None = None
    committed_at: datetime = Field(default_factory=utcnow)


class AuditReport(BaseModel):
    """Per-epoch audit report for independent verifiers (greencompute-audit).

    The canonical on-wire shape is `report_json` with sorted keys + no
    whitespace; `report_sha256` is the hash of that canonical bytes.
    `signature` is the validator's ed25519 signature over the same bytes.
    `chain_commitment_tx` is the tx hash of the Commitments.set_commitment
    extrinsic that anchored `report_sha256` on-chain for this epoch."""

    epoch_id: str = Field(min_length=1, max_length=64)
    netuid: int = Field(ge=0)
    epoch_start_block: int = Field(ge=0)
    epoch_end_block: int = Field(ge=0)
    report_json: dict[str, Any] = Field(default_factory=dict)
    report_sha256: str = Field(min_length=64, max_length=64)
    signature: str = ""
    signer_hotkey: str = ""
    chain_commitment_tx: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
