from enum import StrEnum


class WorkloadKind(StrEnum):
    INFERENCE = "inference"
    POD = "pod"
    VM = "vm"


class SecurityTier(StrEnum):
    STANDARD = "standard"
    CPU_TEE = "cpu_tee"
    CPU_GPU_ATTESTED = "cpu_gpu_attested"


class DeploymentState(StrEnum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PULLING = "pulling"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    SUSPENDED = "suspended"
    FAILED = "failed"
    TERMINATED = "terminated"


class GpuAllocationMode(StrEnum):
    INFERENCE = "inference"
    RENTAL = "rental"
    IDLE = "idle"


class FluxDecision(StrEnum):
    ASSIGN_INFERENCE = "assign_inference"
    ASSIGN_RENTAL = "assign_rental"
    HOLD = "hold"
    QUEUE = "queue"

