"""Load experiment configuration from experiment.toml."""

from __future__ import annotations

import re
from pathlib import Path

from skylab.trial import ExperimentConfig


def load_experiment_config(experiment_dir: Path) -> ExperimentConfig:
    """Parse experiment.toml into an ExperimentConfig."""
    toml_path = experiment_dir / "experiment.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"No experiment.toml found in {experiment_dir}")

    text = toml_path.read_text()
    data = _parse_toml(text)

    config = ExperimentConfig()
    config.experiment_dir = str(experiment_dir.resolve())

    # [experiment] section
    exp = data.get("experiment", {})
    config.name = exp.get("name", experiment_dir.name)
    config.description = exp.get("description", "")

    # [search] section
    search = data.get("search", {})
    config.editable_files = search.get("editable_files", config.editable_files)
    config.frozen_files = search.get("frozen_files", config.frozen_files)
    config.metric = search.get("metric", config.metric)
    config.direction = search.get("direction", config.direction)
    config.metric_pattern = search.get("metric_pattern", config.metric_pattern)

    # [execution] section
    exe = data.get("execution", {})
    config.command = exe.get("command", config.command)
    config.log_file = exe.get("log_file", config.log_file)
    config.time_budget_seconds = int(
        exe.get("time_budget_seconds", config.time_budget_seconds)
    )

    # [constraints] section
    constraints = data.get("constraints", {})
    config.max_vram_gb = float(constraints.get("max_vram_gb", config.max_vram_gb))
    config.max_trial_wall_time = int(
        constraints.get("max_trial_wall_time", config.max_trial_wall_time)
    )

    # Strategy-specific sections (strategy.llm, strategy.sweep, etc.)
    for key, value in data.items():
        if key.startswith("strategy.") or (
            key == "strategy" and isinstance(value, dict)
        ):
            if key == "strategy":
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, dict):
                        config.strategy_config[sub_key] = sub_val
            else:
                strategy_name = key.split(".", 1)[1]
                config.strategy_config[strategy_name] = value

    return config


def _parse_toml(text: str) -> dict:
    """Minimal TOML parser sufficient for experiment.toml.

    Supports: strings, integers, floats, booleans, arrays, dotted sections.
    Does NOT support: inline tables, multi-line strings, datetime, etc.
    """
    result: dict = {}
    current_section: dict = result
    section_path: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section header: [foo] or [foo.bar]
        section_match = re.match(r"^\[([a-zA-Z0-9_.]+)\]$", line)
        if section_match:
            section_path = section_match.group(1).split(".")
            current_section = result
            for part in section_path:
                if part not in current_section:
                    current_section[part] = {}
                current_section = current_section[part]
            continue

        # Key = value
        kv_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$", line)
        if kv_match:
            key = kv_match.group(1)
            raw_value = kv_match.group(2).strip()
            current_section[key] = _parse_value(raw_value)

    return result


def _parse_value(raw: str) -> object:
    """Parse a single TOML value."""
    # String
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")

    # Boolean
    if raw == "true":
        return True
    if raw == "false":
        return False

    # Array (simple, single-line only)
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        items = []
        for item in _split_array(inner):
            items.append(_parse_value(item.strip()))
        return items

    # Integer or float
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _split_array(s: str) -> list[str]:
    """Split array elements respecting nested structures."""
    items = []
    depth = 0
    current = ""
    in_string = False
    for char in s:
        if char == '"' and (not current or current[-1] != "\\"):
            in_string = not in_string
        if not in_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                items.append(current)
                current = ""
                continue
        current += char
    if current.strip():
        items.append(current)
    return items
