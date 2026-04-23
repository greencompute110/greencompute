# greenference

Shared developer-facing code for the Green Compute subnet:

- [`protocol/`](protocol) — cross-service pydantic models, enums, signing helpers, rate constants. Imported by `greenference-api`, `greenference-node`, `greenference-ui` (via TS mirror), and `greenference-audit`.
- [`sdk/`](sdk) — Python client and CLI (`greenference ...`) for programmatic access to the gateway.
- [`examples/`](examples) — SDK example workloads.
- [`tests/`](tests) — protocol + SDK tests.

## protocol (what lives here)

| Module | Contents |
|---|---|
| `models.py` | `WorkloadSpec`, `DeploymentRecord`, `LeaseAssignment`, `NodeCapability`, `ProbeChallenge`/`Result`, `ScoreCard`, `WeightSnapshot`, `ChainWeightCommit`, `AuditReport`, `ModelCatalogEntry`, `CatalogSubmission`, `FluxState`/`RebalanceEvent`, `ChatCompletion{Request,Response,Message,Choice,Usage}`, billing models, ... |
| `enums.py` | `WorkloadKind` (inference/pod/vm), `DeploymentState` (lowercase values), `SecurityTier`, `GpuAllocationMode`, `FluxDecision` |
| `auth.py` | ed25519 / HMAC request signing (`sign_payload_hotkey`, `verify_payload_hotkey`) with replay protection |
| `http_client.py` | `ControlPlaneHTTPClient` — signed HTTP client used by the miner node-agent to talk to the control-plane |
| `billing_rates.py` | Canonical per-GPU hourly rates + `inference_cost_cents(prompt, completion)` shared between gateway pricing and UI estimates |
| `chat.py` | OpenAI-compatible chat completion models — the response shape is pass-through-friendly so vLLM / TGI outputs parse natively |

Everything the validator, gateway, control-plane, miner node-agent, and auditor rely on as a shared contract lives here. A change to `models.py` ripples across every downstream service, so the protocol repo is the first place to edit when adding new fields.

## SDK workflow (optional)

For users who prefer a code-first workflow over the web UI:

1. Define an `Image` and `Workload` in Python.
2. Build from a module ref like `examples/minimal_inference.py:workload`.
3. Create or deploy that workload through the CLI.
4. Inspect builds, deployments, warmup, and utilization from the same CLI.

```bash
greenference config init --base-url https://api.green-compute.com --api-key gk_xxx
greenference build examples/minimal_inference.py:workload --wait
greenference workloads create examples/minimal_inference.py:workload
greenference deploy examples/minimal_inference.py:workload --accept-fee --wait
greenference builds logs <build-id> --follow
greenference deployments wait <deployment-id>
```

## examples

In [`examples/`](examples):

- `minimal_inference.py`
- `vllm_workload.py`
- `diffusion_workload.py`
- `build_only_image.py`

## config

The CLI resolves configuration in this order:

1. CLI flags
2. `GREENFERENCE_API_URL` / `GREENFERENCE_API_KEY` env vars
3. Persisted config under `~/.greenference/config.ini`

```bash
greenference config init --base-url https://api.green-compute.com --api-key gk_xxx
greenference config show
greenference config unset --api-key
```

## tests

```bash
cd greenference
uv sync
pytest
```

## downstream repos

| Repo | How it uses this |
|---|---|
| [`greenference-api`](../greenference-api) | Imports every model in `models.py`; implements services around them |
| [`greenference-node`](../greenference-node) | Uses `ControlPlaneHTTPClient` + miner-side signing |
| [`greenference-audit`](../greenference-audit) | Verifies `AuditReport.signature` against `sign_payload_hotkey` output; ports `ScoreEngine` formula for replay |
| [`greenference-ui`](../greenference-ui) | TypeScript mirrors of model types in [`lib/api/types.ts`](../greenference-ui/lib/api/types.ts) |
