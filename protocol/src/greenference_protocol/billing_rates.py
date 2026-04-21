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
#  Inference — per 1M tokens, in cents
# ---------------------------------------------------------------------------

# Flat rate across all models for now. Per-model tiers (small/medium/large)
# are a follow-up. Values are in cents per 1,000,000 tokens — so 20 = $0.20.
INFERENCE_INPUT_CENTS_PER_MTOK: int = 20   # $0.20 / 1M input tokens
INFERENCE_OUTPUT_CENTS_PER_MTOK: int = 60  # $0.60 / 1M output tokens

# Minimum charge per completion, in cents. Prevents abuse of sub-cent tiny
# requests and covers transport / accounting overhead.
INFERENCE_MIN_CHARGE_CENTS: int = 1


def inference_cost_cents(prompt_tokens: int, completion_tokens: int) -> int:
    """Compute the per-request inference charge in cents, rounded UP to the
    configured minimum. Output tokens are priced higher than input because
    they're what actually runs the model forward."""
    raw_cents = (
        prompt_tokens * INFERENCE_INPUT_CENTS_PER_MTOK
        + completion_tokens * INFERENCE_OUTPUT_CENTS_PER_MTOK
    ) / 1_000_000
    # Round half-up so $0.005 → 1¢, not 0.
    rounded = int(raw_cents + 0.5)
    return max(rounded, INFERENCE_MIN_CHARGE_CENTS)
