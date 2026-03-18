# Greenference SDK Remaining Work Checklist

This checklist covers only `/workspace/Greenference/greenference`.

It does not track broader platform work in:
- `greenference-api`
- `greenference-miner`

## Critical
- [x] Add a real installed-package validation path.
  - [x] Verify `greenference` works from a staged site-packages install path, not only through workspace `PYTHONPATH`.
- [x] Add CLI commands for image/build lifecycle depth.
  - [x] `images history`
  - [x] `builds list`
  - [x] `builds get`
  - [x] `builds logs`
  - [x] `builds wait`
- [x] Add workload lifecycle CLI depth.
  - [x] `workloads create` from code ref without deploy
  - [x] `workloads update` coverage for tags, readme, logo, pricing, alias clearing
  - [x] `workloads utilization`
- [x] Add deployment lifecycle CLI depth.
  - [x] `deployments list`
  - [x] `deployments get`
  - [x] `deployments update`
  - [x] `deployments wait`
- [x] Add stronger error handling and exit behavior.
  - [x] clearer CLI output for HTTP 4xx/5xx
  - [x] non-zero exits for failed build, deploy, and wait flows
- [x] Add packaging size safeguards.
  - [x] warn or fail on oversized local contexts
  - [x] surface included files more clearly before upload

## High
- [x] Expand `Image` DSL parity.
  - [x] maintainer
  - [x] user
  - [x] apt remove
  - [x] richer add/copy semantics
  - [x] better entrypoint/cmd/env ergonomics
- [x] Expand workload DSL structure.
  - [x] explicit concurrency and max instances as first-class SDK fields
  - [x] clearer runtime config object instead of only flat runtime fields
  - [x] support more metadata fields consistently
- [x] Add richer Python client lifecycle helpers.
  - [x] deployment wait helpers with better terminal-state classification
  - [x] workload readiness helpers where meaningful
- [x] Add Python client helpers for shares and warmup that are stable and documented.
- [x] Improve build log streaming UX.
  - [x] consistent SSE parsing
  - [x] explicit end-of-stream and failure states
- [x] Add a real examples directory.
  - [x] minimal inference workload
  - [x] vLLM-style workload
  - [x] diffusion-style workload
  - [x] build-only image example

## Medium
- [x] Normalize template surface.
  - [x] `greenference.templates` should be the canonical API
  - [x] `workloads.py` should remain only as compatibility shim or be removed
- [x] Add typed request/response wrappers in the client.
  - [x] reduce raw dict/list API surface
- [x] Improve config UX.
  - [x] `config unset`
  - [x] `config init`
  - [x] mask secrets in display output
- [x] Add alias-oriented invocation ergonomics.
  - [x] make invoking deployed workloads by alias or derived identifier easier
- [x] Improve packaging controls.
  - [x] ignore patterns
  - [x] explicit include/exclude overrides
  - [x] clearer validation for paths outside project root

## Low
- [x] Expand README and user docs.
  - [x] full code-defined workflow
  - [x] config setup
  - [x] build/deploy/run examples
- [ ] Clean up deprecated helper wording and naming.
  - standardize around `Workload`
- [x] Improve CLI tables and summaries.
  - [x] better status views for builds and deployments

## Tests Required Before Calling The SDK Production-Ready
- [x] Installed-package CLI tests, not only source-tree tests
- [ ] Module-ref build/deploy/run flow against a real local API stack
- [x] Failure-path CLI tests:
  - [x] build failure
  - [x] deploy fee rejection
  - [x] deployment timeout
  - [x] permission denied on share or update
- [x] Packaging tests for nested dirs, ignored files, and large contexts
- [x] Config precedence tests across persisted config, env vars, and CLI flags
- [x] Client retry and timeout tests for build/deploy polling flows

## Production-Ready Definition For This Repo
- [ ] A developer can install `greenference`, define an image/workload in Python, build it, deploy it, inspect it, and invoke it without raw payload assembly.
- [x] The CLI and Python client handle normal failure modes cleanly.
- [x] The packaging flow is safe enough for real project contexts.
- [x] The SDK surface is stable, documented, and tested as an installed package.
