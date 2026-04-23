"""Workload template builders."""

from __future__ import annotations

from greencompute.image import Image
from greencompute.workload import NodeSelector, Workload, WorkloadPack

__all__ = [
    "build_inference_workload",
    "build_vllm_workload",
    "build_diffusion_workload",
]


def build_inference_workload(
    *,
    username: str,
    name: str,
    model_identifier: str,
    runtime_kind: str = "hf-causal-lm",
    image: Image | str | None = None,
    node_selector: NodeSelector | None = None,
    display_name: str | None = None,
    readme: str = "",
    public: bool = False,
    tags: list[str] | None = None,
    workload_alias: str | None = None,
    ingress_host: str | None = None,
    pricing_class: str = "standard",
    logo_uri: str | None = None,
    warmup_enabled: bool = False,
    warmup_path: str | None = None,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    model_revision: str | None = None,
    tokenizer_identifier: str | None = None,
    context_paths: list[str] | None = None,
) -> WorkloadPack:
    resolved_image = image or Image(username=username, name=name, tag="latest", readme=readme)
    workload = Workload(
        name=name,
        image=resolved_image,
        display_name=display_name,
        readme=readme,
        logo_uri=logo_uri,
        public=public,
        tags=tags or [],
        workload_alias=workload_alias,
        ingress_host=ingress_host,
        pricing_class=pricing_class,
        node_selector=node_selector or NodeSelector(),
        runtime_kind=runtime_kind,
        model_identifier=model_identifier,
        model_revision=model_revision,
        tokenizer_identifier=tokenizer_identifier,
        warmup_enabled=warmup_enabled,
        warmup_path=warmup_path,
        scaling_threshold=scaling_threshold,
        shutdown_after_seconds=shutdown_after_seconds,
        context_paths=context_paths or [],
    )
    return WorkloadPack(workload=workload, template="inference")


def build_vllm_workload(
    *,
    username: str,
    name: str,
    model_identifier: str,
    image: Image | str | None = None,
    node_selector: NodeSelector | None = None,
    display_name: str | None = None,
    readme: str = "",
    public: bool = False,
    tags: list[str] | None = None,
    workload_alias: str | None = None,
    ingress_host: str | None = None,
    model_revision: str | None = None,
    tokenizer_identifier: str | None = None,
    context_paths: list[str] | None = None,
) -> WorkloadPack:
    resolved_selector = node_selector or NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24, concurrency=8)
    return build_inference_workload(
        username=username,
        name=name,
        model_identifier=model_identifier,
        runtime_kind="vllm",
        image=image,
        node_selector=resolved_selector,
        display_name=display_name,
        readme=readme,
        public=public,
        tags=tags,
        workload_alias=workload_alias,
        ingress_host=ingress_host,
        warmup_enabled=True,
        warmup_path="/healthz",
        model_revision=model_revision,
        tokenizer_identifier=tokenizer_identifier,
        context_paths=context_paths,
    )


def build_diffusion_workload(
    *,
    username: str,
    name: str,
    model_identifier: str,
    image: Image | str | None = None,
    node_selector: NodeSelector | None = None,
    display_name: str | None = None,
    readme: str = "",
    public: bool = False,
    tags: list[str] | None = None,
    workload_alias: str | None = None,
    ingress_host: str | None = None,
    context_paths: list[str] | None = None,
) -> WorkloadPack:
    resolved_selector = node_selector or NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16, concurrency=1)
    return build_inference_workload(
        username=username,
        name=name,
        model_identifier=model_identifier,
        runtime_kind="diffusion",
        image=image,
        node_selector=resolved_selector,
        display_name=display_name,
        readme=readme,
        public=public,
        tags=tags,
        workload_alias=workload_alias,
        ingress_host=ingress_host,
        context_paths=context_paths,
    )
