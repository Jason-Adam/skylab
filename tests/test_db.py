"""Tests for skylab.db."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from skylab.db import Database
from skylab.trial import TrialResult


def _make_trial(**kwargs: object) -> TrialResult:
    defaults = dict(
        experiment="test-exp",
        commit="abc1234",
        status="success",
        description="test trial",
        val_bpb=1.0,
        peak_vram_mb=40000.0,
        duration_seconds=300.0,
        strategy="sweep",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return TrialResult(**defaults)  # type: ignore[arg-type]


def test_record_and_retrieve() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        trial = _make_trial()
        trial_id = db.record(trial)
        assert trial_id == 1
        assert trial.id == 1

        history = db.history("test-exp")
        assert len(history) == 1
        assert history[0].commit == "abc1234"
        assert history[0].val_bpb == 1.0
        db.close()


def test_update_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        trial = _make_trial()
        db.record(trial)

        assert not db.history("test-exp")[0].kept
        db.update_kept(trial.id, True)  # type: ignore[arg-type]
        assert db.history("test-exp")[0].kept
        db.close()


def test_best_minimizes_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        db.record(_make_trial(val_bpb=1.2, commit="aaa", kept=True))
        db.record(_make_trial(val_bpb=0.9, commit="bbb", kept=True))
        db.record(_make_trial(val_bpb=1.0, commit="ccc", kept=True))

        best = db.best("test-exp")
        assert best is not None
        assert best.val_bpb == 0.9
        assert best.commit == "bbb"
        db.close()


def test_best_excludes_crashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        db.record(_make_trial(val_bpb=None, status="crash", commit="bad", kept=True))
        db.record(_make_trial(val_bpb=1.0, commit="good", kept=True))

        best = db.best("test-exp")
        assert best is not None
        assert best.commit == "good"
        db.close()


def test_latest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        db.record(_make_trial(commit="first"))
        db.record(_make_trial(commit="second"))

        latest = db.latest("test-exp")
        assert latest is not None
        assert latest.commit == "second"
        db.close()


def test_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        assert db.count() == 0
        db.record(_make_trial())
        db.record(_make_trial())
        assert db.count() == 2
        assert db.count("test-exp") == 2
        assert db.count("other") == 0
        db.close()


def test_history_filters_by_experiment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        db.record(_make_trial(experiment="exp-a"))
        db.record(_make_trial(experiment="exp-b"))
        db.record(_make_trial(experiment="exp-a"))

        assert len(db.history("exp-a")) == 2
        assert len(db.history("exp-b")) == 1
        assert len(db.history()) == 3
        db.close()
