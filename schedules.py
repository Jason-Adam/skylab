"""Pure schedule and utility functions used by train.py.

Extracted so they can be imported and tested without a GPU.
"""


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def get_lr_multiplier(
    progress: float,
    warmup_ratio: float,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    """Three-phase LR schedule: warmup -> flat -> cooldown."""
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step: int, ramp_steps: int = 300) -> float:
    """Momentum ramp from 0.85 to 0.95 over ramp_steps."""
    frac = min(step / ramp_steps, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, weight_decay: float) -> float:
    """Linear decay from weight_decay to 0 over training."""
    return weight_decay * (1 - progress)
