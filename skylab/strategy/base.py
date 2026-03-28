"""Strategy protocol — pluggable search strategies."""

from __future__ import annotations

from typing import Protocol

from skylab.trial import ExperimentConfig, Proposal, TrialResult


class Strategy(Protocol):
    """Protocol for experiment search strategies.

    Strategies are stateless — all context comes from the history and
    current code. This makes them swappable mid-run.
    """

    def propose(
        self,
        history: list[TrialResult],
        editable_files: dict[str, str],
        config: ExperimentConfig,
    ) -> Proposal:
        """Given history + current code, propose the next modification.

        Args:
            history: All previous trial results.
            editable_files: Mapping of filename -> current content for editable files.
            config: The experiment configuration.

        Returns:
            A Proposal with modified file contents, description, and rationale.
        """
        ...

    def should_continue(self, history: list[TrialResult]) -> bool:
        """Whether the strategy has more ideas to try."""
        ...
