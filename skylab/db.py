"""SQLite experiment database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from skylab.trial import TrialResult

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment      TEXT    NOT NULL,
    commit_hash     TEXT    NOT NULL,
    parent_commit   TEXT,
    val_bpb         REAL,
    peak_vram_mb    REAL,
    status          TEXT    NOT NULL DEFAULT 'success',
    description     TEXT    NOT NULL DEFAULT '',
    code_diff       TEXT    NOT NULL DEFAULT '',
    duration_seconds REAL   NOT NULL DEFAULT 0.0,
    created_at      TEXT    NOT NULL,
    strategy        TEXT    NOT NULL DEFAULT '',
    kept            INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    """Thin wrapper around a SQLite database for trial tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def record(self, trial: TrialResult) -> int:
        """Insert a trial and return its assigned id."""
        cur = self._conn.execute(
            """\
            INSERT INTO trials
                (experiment, commit_hash, parent_commit, val_bpb, peak_vram_mb,
                 status, description, code_diff, duration_seconds, created_at,
                 strategy, kept)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.experiment,
                trial.commit,
                trial.parent_commit,
                trial.val_bpb,
                trial.peak_vram_mb,
                trial.status,
                trial.description,
                trial.code_diff,
                trial.duration_seconds,
                trial.created_at.isoformat(),
                trial.strategy,
                int(trial.kept),
            ),
        )
        self._conn.commit()
        trial.id = cur.lastrowid
        return cur.lastrowid  # type: ignore[return-value]

    def update_kept(self, trial_id: int, kept: bool) -> None:
        """Update the kept flag on a trial."""
        self._conn.execute(
            "UPDATE trials SET kept = ? WHERE id = ?",
            (int(kept), trial_id),
        )
        self._conn.commit()

    def history(self, experiment: str | None = None) -> list[TrialResult]:
        """Return all trials, optionally filtered by experiment name."""
        if experiment:
            rows = self._conn.execute(
                "SELECT * FROM trials WHERE experiment = ? ORDER BY id",
                (experiment,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM trials ORDER BY id").fetchall()
        return [_row_to_trial(r) for r in rows]

    def best(self, experiment: str, direction: str = "minimize") -> TrialResult | None:
        """Return the best successful trial."""
        order = "ASC" if direction == "minimize" else "DESC"
        row = self._conn.execute(
            f"SELECT * FROM trials WHERE experiment = ? AND status = 'success' "
            f"AND val_bpb IS NOT NULL AND kept = 1 ORDER BY val_bpb {order} LIMIT 1",
            (experiment,),
        ).fetchone()
        return _row_to_trial(row) if row else None

    def latest(self, experiment: str) -> TrialResult | None:
        """Return the most recent trial."""
        row = self._conn.execute(
            "SELECT * FROM trials WHERE experiment = ? ORDER BY id DESC LIMIT 1",
            (experiment,),
        ).fetchone()
        return _row_to_trial(row) if row else None

    def count(self, experiment: str | None = None) -> int:
        """Return the number of trials."""
        if experiment:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trials WHERE experiment = ?",
                (experiment,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()
        return row[0]  # type: ignore[index]

    def close(self) -> None:
        self._conn.close()


def _row_to_trial(row: sqlite3.Row) -> TrialResult:
    from datetime import datetime, timezone

    created = datetime.fromisoformat(row["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return TrialResult(
        id=row["id"],
        experiment=row["experiment"],
        commit=row["commit_hash"],
        parent_commit=row["parent_commit"],
        val_bpb=row["val_bpb"],
        peak_vram_mb=row["peak_vram_mb"],
        status=row["status"],
        description=row["description"],
        code_diff=row["code_diff"],
        duration_seconds=row["duration_seconds"],
        created_at=created,
        strategy=row["strategy"],
        kept=bool(row["kept"]),
    )
