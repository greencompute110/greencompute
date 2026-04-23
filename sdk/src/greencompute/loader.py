"""Module ref loading for Greenference workloads."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from greencompute.workload import Workload, WorkloadPack


@dataclass(slots=True)
class LoadedWorkload:
    module: ModuleType
    module_path: Path
    attribute_name: str
    workload: Workload


def load_workload(module_ref: str) -> LoadedWorkload:
    path_part, sep, attr_name = module_ref.partition(":")
    if not sep or not attr_name:
        raise ValueError("module ref must look like path/to/file.py:workload")
    module_path = Path(path_part).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"module path not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"greencompute_loaded_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, attr_name):
        raise AttributeError(f"module {module_path} has no attribute {attr_name}")
    loaded = getattr(module, attr_name)
    if isinstance(loaded, WorkloadPack):
        workload = loaded.workload
    elif isinstance(loaded, Workload):
        workload = loaded
    else:
        raise TypeError(f"{module_ref} did not resolve to Workload or WorkloadPack")
    return LoadedWorkload(module=module, module_path=module_path, attribute_name=attr_name, workload=workload)
