"""Billing rate constants shared between the gateway and the control-plane.

Kept in the protocol package (rather than a service-specific module) so the
control-plane can consult the same numbers as the gateway without an import
across service boundaries. These are the *source of truth* at runtime —
rates are locked onto each deployment at placement time so that changing
these constants doesn't retroactively affect active rentals.

Change the values here and redeploy all services to roll out a new price.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
#  Rentals / pods — per GPU, per hour, in cents
# ---------------------------------------------------------------------------

# Cents per GPU per hour, keyed by the node's `gpu_model` after
# normalization (see `_normalize_gpu_model` below). Must match the numbers
# published on the /enterprise and landing pages. Keys are the canonical
# human form; `rate_for_gpu` strips dashes/underscores/spaces so real-world
# values like "rtx4090", "rtx-4090", "RTX 4090", "Rtx_4090" all hit the
# same entry.
GPU_RATE_CENTS_PER_HOUR: dict[str, int] = {
    "rtx4090": 40,  # $0.40/hr/GPU
    "rtx5090": 70,  # $0.70/hr/GPU
}

# Fallback for any GPU model that isn't in the table above (legacy H100/A100
# rentals etc.). Kept at the old flat $0.10/hr so pre-existing rentals aren't
# suddenly charged at a different rate when the new code ships.
LEGACY_FALLBACK_CENTS_PER_HOUR: int = 10


# Canonical hardware specs for known GPUs — used to correct capacity
# updates from miners whose env var was set incorrectly (e.g. a 5090 host
# reporting 24GB VRAM because the operator mis-configured
# GREENCOMPUTE_VRAM_GB_PER_GPU). Keyed by normalized model name.
GPU_VRAM_GB: dict[str, int] = {
    "rtx4090": 24,
    "rtx5090": 32,
}


def canonical_vram_gb(gpu_model: str | None, reported: int | None = None) -> int | None:
    """Return the known-good VRAM for a GPU model, falling back to the
    miner-reported value if the model is unknown. `None` if both are
    missing."""
    if gpu_model:
        canonical = GPU_VRAM_GB.get(_normalize_gpu_model(gpu_model))
        if canonical is not None:
            return canonical
    return reported


def _normalize_gpu_model(raw: str) -> str:
    """Lower-case and strip separators so callers don't have to care about
    whether their GPU id is "rtx-4090" / "rtx_4090" / "RTX 4090" / "rtx4090"."""
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def rate_for_gpu(gpu_model: str | None) -> int:
    """Return the per-GPU-per-hour rate in cents for a given GPU model.
    Returns the legacy fallback for unknown / missing models."""
    if not gpu_model:
        return LEGACY_FALLBACK_CENTS_PER_HOUR
    return GPU_RATE_CENTS_PER_HOUR.get(
        _normalize_gpu_model(gpu_model),
        LEGACY_FALLBACK_CENTS_PER_HOUR,
    )


# ---------------------------------------------------------------------------
#  Saved ("archived") pods — storage hold while suspended, per pod per day
# ---------------------------------------------------------------------------

# Flat charge per SAVED pod per day while it sits SUSPENDED with the user's
# work preserved (the opt-in "save my pod" choice). A single flat rate per pod
# per day — NOT per-GB — is what sales agreed. Billed as a negative-balance
# debt (debit with allow_negative) so a later top-up settles it automatically
# via ordinary balance arithmetic. Capped by `pod_saved_retention_days` so the
# worst-case debt is bounded (30 days → $30).
STORAGE_CENTS_PER_DAY: int = 100  # $1.00 / pod / day

_MINUTES_PER_DAY = 1440


def storage_mcents_per_minute(cents_per_day: int = STORAGE_CENTS_PER_DAY) -> int:
    """Per-minute storage charge in MILLICENTS, matching the metering loop's
    per-minute accrual cadence. The sub-cent remainder is carried on each
    deployment's `storage_remainder_mcents` accumulator (like
    `metering_remainder_mcents`), so it converges to exactly `cents_per_day`
    over any 24h window regardless of tick jitter."""
    return round(cents_per_day * 1000 / _MINUTES_PER_DAY)


# ---------------------------------------------------------------------------
#  Inference — per 1M tokens, in cents
# ---------------------------------------------------------------------------

# DEFAULT rate, applied to any model without an explicit entry below. Sized for
# 7B-class models that run on ONE GPU: a 5090 costs $0.70/hr and serves >1k
# tok/s, so $0.60/1M output is roughly a 3x margin. Values are in cents per
# 1,000,000 tokens — so 20 = $0.20.
INFERENCE_INPUT_CENTS_PER_MTOK: int = 20   # $0.20 / 1M input tokens
INFERENCE_OUTPUT_CENTS_PER_MTOK: int = 60  # $0.60 / 1M output tokens

# PER-MODEL overrides: model_id -> (input_cents_per_mtok, output_cents_per_mtok).
#
# The default above is a per-GPU-economics number and does NOT survive contact
# with a model that pins a whole cluster. Kimi K3 spans 72 GPUs (9 nodes x 8
# RTX 5090) for ONE replica, so leaving it on the default billed $0.60/1M
# output against a measured cost of ~$162/1M.
#
# K3 is priced DELIBERATELY BELOW the market. Reference price across all ~11
# commercial K3 endpoints is $3.00/$15.00 (Moonshot's own list price, copied
# verbatim); the cheapest single endpoint is Morph at $2.90/$14.00. We sit
# ~33% under the reference at $2.00/$10.00, which makes GreenCompute the
# cheapest K3 anywhere and reflects the hardware story: 72x RTX 5090 at
# $0.70/GPU-hr vs an 8xB300 node at $65+/hr, on 100% renewable power.
#
# Known and accepted: at the measured 86 tok/s aggregate ceiling every K3 token
# is served below cost either way (~$162/1M at the $50.40/hr opportunity cost of
# the 72 cards), so the discount changes the economics by only ~3% — the
# throughput is the cost driver, not the price. This is a positioning decision,
# not a margin one. Revisit if aggregate throughput ever clears ~930 tok/s,
# which is break-even at the $15 reference.
# See project_k3_pricing_economics for the full derivation.
MODEL_RATE_CENTS_PER_MTOK: dict[str, tuple[int, int]] = {
    "kimi-k3": (200, 1000),
    # K3 CODER (REAP-320) on 48x RTX 5090. Pruned to ~1.03T params so it is
    # cheaper to serve than full K3, but still pins a whole 6-node cluster.
    "k3-coder": (100, 500),  # $1.00 in / $5.00 out  # $2.00 / 1M in, $10.00 / 1M out
}

# Minimum charge per completion, in cents. Prevents abuse of sub-cent tiny
# requests and covers transport / accounting overhead.
INFERENCE_MIN_CHARGE_CENTS: int = 1


def _normalize_model_id(model: str | None) -> str:
    """Match a rate entry regardless of how the caller spells the model.

    Clients may send the catalog id (`kimi-k3`) or the HF repo
    (`moonshotai/Kimi-K3`); both must bill the same, or the vendor-prefixed
    form silently falls through to the cheap default.
    """
    if not model:
        return ""
    name = model.strip().lower()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def rates_for_model(model: str | None) -> tuple[int, int]:
    """(input, output) cents per 1M tokens for `model`, falling back to the
    flat default for anything without an explicit entry."""
    return MODEL_RATE_CENTS_PER_MTOK.get(
        _normalize_model_id(model),
        (INFERENCE_INPUT_CENTS_PER_MTOK, INFERENCE_OUTPUT_CENTS_PER_MTOK),
    )


def inference_cost_cents(
    prompt_tokens: int,
    completion_tokens: int,
    model: str | None = None,
) -> int:
    """Compute the per-request inference charge in cents, rounded UP to the
    configured minimum. Output tokens are priced higher than input because
    they're what actually runs the model forward.

    `model` is optional for backward compatibility; omitting it bills at the
    default rate, which is correct for single-GPU models and WRONG for a
    cluster-scale one — always pass it where the model is known.
    """
    input_rate, output_rate = rates_for_model(model)
    raw_cents = (
        prompt_tokens * input_rate
        + completion_tokens * output_rate
    ) / 1_000_000
    # Round half-up so $0.005 → 1¢, not 0.
    rounded = int(raw_cents + 0.5)
    return max(rounded, INFERENCE_MIN_CHARGE_CENTS)
