"""Context summarization for LLM strategy (AIDE pattern).

Compresses trial history into a bounded context window so the LLM
can make informed decisions without seeing the full log.
"""

from __future__ import annotations

from skylab.trial import TrialResult


def summarize_history(
    history: list[TrialResult],
    max_trials: int = 20,
) -> str:
    """Summarize trial history into a compact text representation.

    Shows the most recent trials and the overall best result.
    Kept trials are marked with [KEPT], reverted with [REVERTED].
    """
    if not history:
        return "No experiments run yet."

    lines: list[str] = []

    # Overall stats
    successful = [t for t in history if t.status == "success" and t.val_bpb is not None]
    kept = [t for t in history if t.kept]

    lines.append(f"## Experiment History ({len(history)} trials total)")
    lines.append("")

    if kept:
        best_kept = min(kept, key=lambda t: t.val_bpb or float("inf"))
        lines.append(f"**Best result so far**: val_bpb = {best_kept.val_bpb:.6f}")
        lines.append(f"  Achieved by: {best_kept.description}")
        lines.append("")

    # Show recent trials (most relevant for decision-making)
    recent = history[-max_trials:]
    if len(history) > max_trials:
        lines.append(f"(showing last {max_trials} of {len(history)} trials)")
        lines.append("")

    lines.append("| # | Status | val_bpb | VRAM (MB) | Kept | Description |")
    lines.append("|---|--------|---------|-----------|------|-------------|")

    for trial in recent:
        num = trial.id or "?"
        status = trial.status
        bpb = f"{trial.val_bpb:.6f}" if trial.val_bpb is not None else "N/A"
        vram = f"{trial.peak_vram_mb:.0f}" if trial.peak_vram_mb is not None else "N/A"
        kept_str = "KEPT" if trial.kept else "reverted"
        desc = trial.description[:60]
        lines.append(f"| {num} | {status} | {bpb} | {vram} | {kept_str} | {desc} |")

    lines.append("")

    # Highlight patterns
    if len(successful) >= 3:
        improvements = [t for t in successful if t.kept]
        regressions = [t for t in successful if not t.kept]
        crashes = [t for t in history if t.status in ("crash", "timeout")]

        if improvements:
            lines.append("**What worked:**")
            for t in improvements[-5:]:
                lines.append(f"  - {t.description} (val_bpb={t.val_bpb:.6f})")
            lines.append("")

        if regressions:
            lines.append("**What didn't work:**")
            for t in regressions[-5:]:
                lines.append(f"  - {t.description} (val_bpb={t.val_bpb:.6f})")
            lines.append("")

        if crashes:
            lines.append("**Crashes:**")
            for t in crashes[-3:]:
                lines.append(f"  - {t.description} ({t.status})")
            lines.append("")

    return "\n".join(lines)
