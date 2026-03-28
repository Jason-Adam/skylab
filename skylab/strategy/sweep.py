"""Hyperparameter sweep strategy — grid/random search over module-level constants."""

from __future__ import annotations

import itertools
import random
import re

from skylab.trial import ExperimentConfig, Proposal, TrialResult

# Matches lines like: DEPTH = 8, MATRIX_LR = 0.04, WEIGHT_DECAY = 0.2
CONSTANT_RE = re.compile(r"^([A-Z_]+)\s*=\s*(.+?)(\s*#.*)?$", re.MULTILINE)


class SweepStrategy:
    """Grid or random search over module-level constants in train.py.

    Reads parameter definitions from experiment.toml [strategy.sweep].
    Each parameter specifies either explicit values or a range.
    """

    def __init__(self, sweep_config: dict) -> None:
        self._params = sweep_config.get("parameters", [])
        self._mode = sweep_config.get("mode", "grid")  # "grid" or "random"
        self._max_trials = sweep_config.get("max_trials", 100)

        # Pre-compute grid if in grid mode
        if self._mode == "grid":
            self._grid = self._build_grid()
            random.shuffle(self._grid)  # Randomize grid order
        else:
            self._grid = []

        self._trial_index = 0

    def propose(
        self,
        history: list[TrialResult],
        editable_files: dict[str, str],
        config: ExperimentConfig,
    ) -> Proposal:
        if self._mode == "grid":
            combo = self._grid[self._trial_index]
        else:
            combo = self._random_sample()

        self._trial_index += 1

        # Apply the parameter combination to train.py
        train_py = editable_files.get("train.py", "")
        modified = _apply_constants(train_py, combo)
        if modified == train_py:
            raise ValueError(
                f"Sweep produced no changes for {combo}. "
                "Ensure parameter names match module-level constants in train.py."
            )

        desc_parts = [f"{k}={v}" for k, v in combo.items()]
        description = "sweep: " + ", ".join(desc_parts)

        return Proposal(
            description=description,
            rationale=f"Systematic sweep ({self._mode} mode), trial {self._trial_index}",
            modified_files={"train.py": modified},
        )

    def should_continue(self, history: list[TrialResult]) -> bool:
        if self._trial_index >= self._max_trials:
            return False
        if self._mode == "grid" and self._trial_index >= len(self._grid):
            return False
        return True

    def _build_grid(self) -> list[dict[str, object]]:
        """Build the full grid of parameter combinations."""
        names: list[str] = []
        value_lists: list[list[object]] = []
        for param in self._params:
            names.append(param["name"])
            if "values" in param:
                value_lists.append(param["values"])
            elif "range" in param:
                lo, hi = param["range"]
                steps = param.get("steps", 5)
                log_scale = param.get("log_scale", False)
                value_lists.append(_linspace(lo, hi, steps, log_scale))

        return [
            dict(zip(names, combo, strict=False))
            for combo in itertools.product(*value_lists)
        ]

    def _random_sample(self) -> dict[str, object]:
        """Sample a random parameter combination."""
        combo: dict[str, object] = {}
        for param in self._params:
            name = param["name"]
            if "values" in param:
                combo[name] = random.choice(param["values"])
            elif "range" in param:
                lo, hi = param["range"]
                log_scale = param.get("log_scale", False)
                if log_scale:
                    import math

                    combo[name] = math.exp(random.uniform(math.log(lo), math.log(hi)))
                else:
                    combo[name] = random.uniform(lo, hi)
                # If original values are ints, keep as int
                if isinstance(lo, int) and isinstance(hi, int):
                    combo[name] = int(round(combo[name]))  # type: ignore[arg-type]
        return combo


def _apply_constants(source: str, constants: dict[str, object]) -> str:
    """Replace module-level constant assignments in Python source."""

    def replacer(match: re.Match) -> str:
        name = match.group(1)
        if name in constants:
            value = constants[name]
            comment = match.group(3) or ""
            return f"{name} = {_format_value(value)}{comment}"
        return match.group(0)

    return CONSTANT_RE.sub(replacer, source)


def _format_value(value: object) -> str:
    """Format a Python value for source code."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _linspace(lo: float, hi: float, steps: int, log_scale: bool) -> list[float]:
    """Generate evenly spaced values, optionally on a log scale."""
    import math

    if steps <= 1:
        return [lo]
    if log_scale:
        log_lo, log_hi = math.log(lo), math.log(hi)
        return [
            math.exp(log_lo + i * (log_hi - log_lo) / (steps - 1)) for i in range(steps)
        ]
    return [lo + i * (hi - lo) / (steps - 1) for i in range(steps)]
