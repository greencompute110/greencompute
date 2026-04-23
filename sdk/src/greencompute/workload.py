"""Developer-facing workload definitions for GreenCompute."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from greencompute.image import Image


@dataclass(slots=True)
class NodeSelector:
    gpu_count: int = 1
    min_vram_gb_per_gpu: int = 16
    include: list[str] | None = None
    exclude: list[str] | None = None
    cpu_cores: int = 8
    memory_gb: int = 32
    concurrency: int = 1
    max_instances: int = 1

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be >= 1")
        if self.min_vram_gb_per_gpu < 1:
            raise ValueError("min_vram_gb_per_gpu must be >= 1")
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")
        if self.memory_gb < 1:
            raise ValueError("memory_gb must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.max_instances < 1:
            raise ValueError("max_instances must be >= 1")

    def to_requirements_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gpu_count": self.gpu_count,
            "min_vram_gb_per_gpu": self.min_vram_gb_per_gpu,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "concurrency": self.concurrency,
            "max_instances": self.max_instances,
        }
        if self.include:
            payload["supported_gpu_models"] = self.include
        return payload


@dataclass(slots=True)
class RuntimeConfig:
    runtime_kind: str = "hf-causal-lm"
    model_identifier: str = "sshleifer/tiny-gpt2"
    model_revision: str | None = None
    tokenizer_identifier: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "runtime_kind": self.runtime_kind,
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "tokenizer_identifier": self.tokenizer_identifier,
        }


@dataclass(slots=True)
class Workload:
    name: str
    image: str | Image
    node_selector: NodeSelector = field(default_factory=NodeSelector)
    runtime: RuntimeConfig | None = None
    display_name: str | None = None
    tagline: str = ""
    readme: str = ""
    logo_uri: str | None = None
    tags: list[str] = field(default_factory=list)
    workload_alias: str | None = None
    ingress_host: str | None = None
    pricing_class: str = "standard"
    runtime_kind: str = "hf-causal-lm"
    model_identifier: str = "sshleifer/tiny-gpt2"
    model_revision: str | None = None
    tokenizer_identifier: str | None = None
    scaling_threshold: float = 0.75
    shutdown_after_seconds: int = 300
    warmup_enabled: bool = False
    warmup_path: str | None = None
    public: bool = False
    kind: str = "inference"
    security_tier: str = "standard"
    context_paths: list[str] = field(default_factory=list)
    include_paths: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.include_paths:
            self.context_paths = [*self.context_paths, *self.include_paths]
        if self.runtime is None:
            self.runtime = RuntimeConfig(
                runtime_kind=self.runtime_kind,
                model_identifier=self.model_identifier,
                model_revision=self.model_revision,
                tokenizer_identifier=self.tokenizer_identifier,
            )
        else:
            self.runtime_kind = self.runtime.runtime_kind
            self.model_identifier = self.runtime.model_identifier
            self.model_revision = self.runtime.model_revision
            self.tokenizer_identifier = self.runtime.tokenizer_identifier

    @property
    def image_ref(self) -> str:
        if isinstance(self.image, Image):
            return self.image.reference
        return self.image

    @property
    def invocation_model(self) -> str:
        return self.workload_alias or self.name

    def to_build_payload(
        self,
        *,
        context_uri: str,
        public: bool | None = None,
    ) -> dict[str, Any]:
        return {
            "image": self.image_ref,
            "context_uri": context_uri,
            "dockerfile_path": "Dockerfile",
            "display_name": self.display_name,
            "readme": self.readme,
            "logo_uri": self.logo_uri,
            "tags": self.tags,
            "public": self.public if public is None else public,
        }

    def to_workload_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image": self.image_ref,
            "display_name": self.display_name,
            "readme": self.readme,
            "logo_uri": self.logo_uri,
            "tags": self.tags,
            "workload_alias": self.workload_alias,
            "ingress_host": self.ingress_host,
            "kind": self.kind,
            "security_tier": self.security_tier,
            "pricing_class": self.pricing_class,
            "requirements": self.node_selector.to_requirements_payload(),
            "runtime": self.runtime.to_payload() if self.runtime is not None else RuntimeConfig().to_payload(),
            "lifecycle": {
                "scaling_threshold": self.scaling_threshold,
                "shutdown_after_seconds": self.shutdown_after_seconds,
                "warmup_enabled": self.warmup_enabled,
                "warmup_path": self.warmup_path,
            },
            "public": self.public,
        }

    def to_deployment_payload(self, *, requested_instances: int = 1, accept_fee: bool = False) -> dict[str, Any]:
        return {
            "requested_instances": requested_instances,
            "accept_fee": accept_fee,
        }


@dataclass(slots=True)
class WorkloadPack:
    workload: Workload
    template: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
