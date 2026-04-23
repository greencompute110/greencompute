"""Declarative image DSL for Greenference."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import NAMESPACE_OID, uuid5


def _validate_name(value: str, field_name: str) -> str:
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", value, re.I):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return value


@dataclass(slots=True)
class _Directive:
    line: str
    context_paths: tuple[str, ...] = ()


@dataclass(slots=True)
class Image:
    username: str
    name: str
    tag: str
    readme: str = ""
    base_image: str = "python:3.12-slim"
    _directives: list[_Directive] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.username = _validate_name(self.username, "username")
        self.name = _validate_name(self.name, "image name")
        self.tag = _validate_name(self.tag, "image tag")
        self._directives.insert(0, _Directive(f"FROM {self.base_image}"))

    @property
    def uid(self) -> str:
        return str(uuid5(NAMESPACE_OID, f"{self.username}/{self.name}:{self.tag}".lower()))

    @property
    def reference(self) -> str:
        return f"{self.username}/{self.name}:{self.tag}"

    @property
    def build_context_paths(self) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for directive in self._directives:
            for source in directive.context_paths:
                normalized = str(Path(source))
                if normalized in seen:
                    continue
                seen.add(normalized)
                paths.append(normalized)
        return paths

    def __str__(self) -> str:
        return "\n".join(directive.line for directive in self._directives)

    def from_base(self, base_image: str) -> Image:
        self.base_image = base_image
        self._directives[0] = _Directive(f"FROM {base_image}")
        return self

    def with_env(self, key: str, value: str) -> Image:
        self._directives.append(_Directive(f"ENV {key}={value}"))
        return self

    def with_maintainer(self, value: str) -> Image:
        self._directives.append(_Directive(f"LABEL maintainer={value!r}"))
        return self

    def as_user(self, user: str) -> Image:
        self._directives.append(_Directive(f"USER {user}"))
        return self

    def apt_install(self, packages: str | Iterable[str]) -> Image:
        joined = packages if isinstance(packages, str) else " ".join(packages)
        self._directives.append(
            _Directive(
                "RUN apt-get update && apt-get install -y "
                f"{joined} && rm -rf /var/lib/apt/lists/*"
            )
        )
        return self

    def apt_remove(self, packages: str | Iterable[str]) -> Image:
        joined = packages if isinstance(packages, str) else " ".join(packages)
        self._directives.append(_Directive(f"RUN apt-get remove -y {joined} && apt-get autoremove -y"))
        return self

    def run_command(self, command: str) -> Image:
        self._directives.append(_Directive(f"RUN {command}"))
        return self

    def add(self, source: str, destination: str) -> Image:
        self._directives.append(_Directive(f"ADD {source} {destination}", (source,)))
        return self

    def copy(self, source: str, destination: str) -> Image:
        self._directives.append(_Directive(f"COPY {source} {destination}", (source,)))
        return self

    def set_workdir(self, directory: str) -> Image:
        self._directives.append(_Directive(f"WORKDIR {directory}"))
        return self

    def with_entrypoint(self, *args: str) -> Image:
        quoted = ", ".join(f'"{item}"' for item in args)
        self._directives.append(_Directive(f"ENTRYPOINT [{quoted}]"))
        return self

    def with_cmd(self, *args: str) -> Image:
        quoted = ", ".join(f'"{item}"' for item in args)
        self._directives.append(_Directive(f"CMD [{quoted}]"))
        return self
