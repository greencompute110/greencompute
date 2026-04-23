"""Compatibility wrappers for Greenference workload templates."""

from __future__ import annotations

import warnings

from greencompute.templates import build_diffusion_workload, build_vllm_workload, build_inference_workload

__all__ = ["create_vllm_workload", "create_diffusion_workload", "create_inference_workload"]


def create_vllm_workload(*args, **kwargs):
    warnings.warn(
        "greencompute.workloads.create_vllm_workload is deprecated; use greencompute.templates.build_vllm_workload",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_vllm_workload(*args, **kwargs)


def create_diffusion_workload(*args, **kwargs):
    warnings.warn(
        "greencompute.workloads.create_diffusion_workload is deprecated; use greencompute.templates.build_diffusion_workload",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_diffusion_workload(*args, **kwargs)


def create_inference_workload(*args, **kwargs):
    warnings.warn(
        "greencompute.workloads.create_inference_workload is deprecated; use greencompute.templates.build_inference_workload",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_inference_workload(*args, **kwargs)
