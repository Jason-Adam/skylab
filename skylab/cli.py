"""Skylab CLI — autonomous pretraining research tool."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="skylab",
        description="Autonomous pretraining research tool",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # --- skylab run ---
    run_parser = sub.add_parser("run", help="Run the experiment loop")
    run_parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiments/gpt-pretrain",
        help="Path to experiment directory (default: experiments/gpt-pretrain)",
    )
    run_parser.add_argument(
        "--strategy",
        choices=["llm", "sweep"],
        default="llm",
        help="Search strategy (default: llm)",
    )
    run_parser.add_argument(
        "--runner",
        choices=["local", "remote"],
        default="local",
        help="Execution backend (default: local)",
    )
    run_parser.add_argument(
        "--budget",
        type=str,
        default=None,
        help="Time budget (e.g. '8h', '30m'). Default: unlimited",
    )
    run_parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials. Default: strategy decides",
    )
    run_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Git branch tag (creates skylab/<tag> branch)",
    )
    run_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )

    # --- skylab history ---
    hist_parser = sub.add_parser("history", help="Show experiment history")
    hist_parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiments/gpt-pretrain",
    )
    hist_parser.add_argument("--best", action="store_true", help="Show best trial only")
    hist_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export to TSV file (backwards compat with results.tsv)",
    )

    # --- skylab monitor ---
    mon_parser = sub.add_parser("monitor", help="Launch experiment dashboard")
    mon_parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiments/gpt-pretrain",
    )
    mon_parser.add_argument("--port", type=int, default=8080)
    mon_parser.add_argument(
        "--report", action="store_true", help="Generate static report"
    )

    # --- skylab prepare ---
    prep_parser = sub.add_parser("prepare", help="Prepare data and tokenizer")
    prep_parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiments/gpt-pretrain",
    )

    # --- skylab check-gpu ---
    gpu_parser = sub.add_parser("check-gpu", help="Check GPU availability")
    gpu_parser.add_argument(
        "--runner",
        choices=["local", "remote"],
        default="local",
        help="Which runner to check (default: local)",
    )
    gpu_parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiments/gpt-pretrain",
    )

    args = parser.parse_args(argv)

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "history":
        _cmd_history(args)
    elif args.command == "monitor":
        _cmd_monitor(args)
    elif args.command == "prepare":
        _cmd_prepare(args)
    elif args.command == "check-gpu":
        _cmd_check_gpu(args)


def _cmd_run(args: argparse.Namespace) -> None:
    import os

    from skylab.config import load_experiment_config
    from skylab.db import Database
    from skylab.orchestrator import run

    # Set device for training subprocess to pick up (local runner only)
    if args.device != "auto" and args.runner == "local":
        os.environ["SKYLAB_DEVICE"] = args.device

    experiment_dir = Path(args.experiment_dir)
    config = load_experiment_config(experiment_dir)
    db = Database(experiment_dir / "skylab.db")

    # Build strategy
    if args.strategy == "llm":
        from skylab.strategy.llm import LLMStrategy

        llm_config = config.strategy_config.get("llm", {})
        strategy = LLMStrategy(llm_config)  # type: ignore[arg-type]
    elif args.strategy == "sweep":
        from skylab.strategy.sweep import SweepStrategy

        sweep_config = config.strategy_config.get("sweep", {})
        strategy = SweepStrategy(sweep_config)  # type: ignore[arg-type]
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)

    # Build runner
    if args.runner == "local":
        from skylab.runner.local import LocalRunner

        runner = LocalRunner()
    elif args.runner == "remote":
        from skylab.runner.remote import RemoteRunner, load_remote_config

        remote_config = load_remote_config(experiment_dir)
        runner = RemoteRunner(remote_config)
    else:
        print(f"Unknown runner: {args.runner}", file=sys.stderr)
        sys.exit(1)

    # Parse budget
    budget_hours = _parse_budget(args.budget) if args.budget else None

    try:
        best = run(
            experiment_dir=experiment_dir,
            strategy=strategy,
            runner=runner,
            db=db,
            budget_hours=budget_hours,
            max_trials=args.max_trials,
            tag=args.tag,
        )
        if best:
            bpb = f"{best.val_bpb:.6f}" if best.val_bpb is not None else "N/A"
            print(f"\nBest result: val_bpb = {bpb} (commit {best.commit})")
        else:
            print("\nNo successful trials.", file=sys.stderr)
            sys.exit(1)
    finally:
        db.close()


def _cmd_history(args: argparse.Namespace) -> None:
    from skylab.config import load_experiment_config
    from skylab.db import Database

    experiment_dir = Path(args.experiment_dir)
    db_path = experiment_dir / "skylab.db"

    if not db_path.exists():
        print("No experiment database found.", file=sys.stderr)
        sys.exit(1)

    config = load_experiment_config(experiment_dir)
    db = Database(db_path)

    try:
        if args.best:
            best = db.best(config.name, config.direction)
            if best:
                _print_trial(best)
            else:
                print("No successful trials.")
            return

        if args.export:
            _export_tsv(db, config.name, args.export)
            return

        # Print full history
        history = db.history(config.name)
        if not history:
            print("No trials recorded.")
            return

        # Header
        print(
            f"{'#':>4}  {'Status':>8}  {'val_bpb':>10}  {'VRAM MB':>8}  {'Kept':>6}  Description"
        )
        print("-" * 80)

        for trial in history:
            num = trial.id or "?"
            bpb = f"{trial.val_bpb:.6f}" if trial.val_bpb is not None else "N/A"
            vram = (
                f"{trial.peak_vram_mb:.0f}" if trial.peak_vram_mb is not None else "N/A"
            )
            kept = "KEPT" if trial.kept else ""
            desc = trial.description[:40]
            print(
                f"{num:>4}  {trial.status:>8}  {bpb:>10}  {vram:>8}  {kept:>6}  {desc}"
            )
    finally:
        db.close()


def _cmd_monitor(args: argparse.Namespace) -> None:
    import os

    # monitor.py reads run.log and results.tsv from cwd
    experiment_dir = Path(args.experiment_dir).resolve()
    os.chdir(str(experiment_dir))

    from skylab.monitor.server import generate_report, serve

    if args.report:
        generate_report(experiment_dir / "report.html")
    else:
        serve(args.port)


def _cmd_prepare(args: argparse.Namespace) -> None:
    import subprocess

    experiment_dir = Path(args.experiment_dir)
    subprocess.run(
        ["uv", "run", "prepare.py"],
        cwd=str(experiment_dir),
    )


def _cmd_check_gpu(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)

    if args.runner == "remote":
        from skylab.runner.remote import RemoteRunner, load_remote_config

        remote_config = load_remote_config(experiment_dir)
        runner = RemoteRunner(remote_config)
    else:
        from skylab.runner.local import LocalRunner

        runner = LocalRunner()  # type: ignore[assignment]

    if runner.check():
        if args.runner == "local":
            print(f"Device available: {runner.detected_device}")
        else:
            print("GPU available.")
    else:
        print("GPU not available.", file=sys.stderr)
        sys.exit(1)


def _print_trial(trial) -> None:
    print(f"Trial #{trial.id}")
    print(f"  Status:      {trial.status}")
    print(
        f"  val_bpb:     {trial.val_bpb:.6f}"
        if trial.val_bpb is not None
        else "  val_bpb:     N/A"
    )
    print(
        f"  VRAM (MB):   {trial.peak_vram_mb:.0f}"
        if trial.peak_vram_mb is not None
        else "  VRAM (MB):   N/A"
    )
    print(f"  Commit:      {trial.commit}")
    print(f"  Strategy:    {trial.strategy}")
    print(f"  Duration:    {trial.duration_seconds:.1f}s")
    print(f"  Description: {trial.description}")
    print(f"  Kept:        {trial.kept}")


def _export_tsv(db, experiment: str, path: str) -> None:
    history = db.history(experiment)
    with open(path, "w") as f:
        f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        for trial in history:
            commit = trial.commit or ""
            bpb = f"{trial.val_bpb:.6f}" if trial.val_bpb is not None else "0.000000"
            mem_gb = f"{trial.peak_vram_mb / 1024:.1f}" if trial.peak_vram_mb else "0.0"
            status = (
                "keep"
                if trial.kept
                else ("crash" if trial.status == "crash" else "discard")
            )
            desc = trial.description
            f.write(f"{commit}\t{bpb}\t{mem_gb}\t{status}\t{desc}\n")
    print(f"Exported {len(history)} trials to {path}")


def _parse_budget(budget_str: str) -> float:
    """Parse budget string like '8h', '30m', '2d' into hours."""
    budget_str = budget_str.strip().lower()
    if budget_str.endswith("h"):
        return float(budget_str[:-1])
    if budget_str.endswith("m"):
        return float(budget_str[:-1]) / 60
    if budget_str.endswith("d"):
        return float(budget_str[:-1]) * 24
    return float(budget_str)


if __name__ == "__main__":
    main()
