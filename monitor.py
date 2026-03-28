"""Lightweight HTML dashboard for monitoring ML experiment runs.

Usage:
    uv run monitor.py               # Start on localhost:8080
    uv run monitor.py --port 9090   # Custom port
    uv run monitor.py --report      # Generate report.html from history and exit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

WORKDIR = Path(__file__).parent


@dataclass
class RunState:
    status: str = "idle"  # "training" | "completed" | "idle" | "no_data"
    step: int = 0
    progress_pct: float = 0.0
    loss: float = 0.0
    lr_mult: float = 0.0
    ms_per_step: float = 0.0
    tok_per_sec: float = 0.0
    mfu: float = 0.0
    epoch: int = 0
    remaining: str = ""
    summary: dict[str, str] = field(default_factory=dict)


@dataclass
class GpuInfo:
    name: str
    utilization: int
    memory_used: int
    memory_total: int
    temperature: int
    power: float


@dataclass
class RemoteConfig:
    host: str
    user: str
    key_path: str
    workspace: str = "/workspace/skylab"
    num_gpus: int = 1


@dataclass
class DashboardState:
    run: RunState
    gpus: list[GpuInfo]
    history: list[dict[str, str]]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# Matches: step  142 | 47.3% | loss 3.421 | lr_mult 0.83 | 234.5 ms/step | 12345 tok/sec | mfu 42.1% | epoch 1 | 158.3s remaining
_PROGRESS_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*([\d.]+)%\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr_mult\s+([\d.]+)"
    r"\s*\|\s*([\d.]+)\s*ms/step\s*\|\s*([\d.]+)\s*tok/sec\s*\|\s*mfu\s*([\d.]+)%"
    r"\s*\|\s*epoch\s+(\d+)\s*\|\s*([\d.]+s\s*remaining)"
)


def parse_run_log(path: Path) -> RunState:
    """Parse run.log and return a RunState."""
    if not path.exists():
        return RunState(status="no_data")

    age = time.time() - path.stat().st_mtime
    # Read raw bytes to preserve bare \r used as progress-bar line delimiter.
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")

    # Detect summary block (lines after "---" sentinel).
    # The summary block uses normal \n line endings.
    summary: dict[str, str] = {}
    if "---" in text:
        after = text.split("---", 1)[1]
        for line in after.splitlines():
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                summary[k.strip()] = v.strip()

    # Find the last \r-delimited progress line.
    # Split on bare \r (i.e. b'\r' that is NOT followed by \n).
    # We work on the raw bytes then decode each chunk.
    raw_chunks = re.split(rb"\r(?!\n)", raw)
    progress_line: str | None = None
    for raw_chunk in reversed(raw_chunks):
        chunk = raw_chunk.decode("utf-8", errors="replace").strip()
        if chunk:
            progress_line = chunk
            break

    state = RunState(summary=summary)

    if summary:
        state.status = "completed"
    elif age > 30:
        state.status = "idle"
    else:
        state.status = "training"

    if progress_line:
        m = _PROGRESS_RE.search(progress_line)
        if m:
            state.step = int(m.group(1))
            state.progress_pct = float(m.group(2))
            state.loss = float(m.group(3))
            state.lr_mult = float(m.group(4))
            state.ms_per_step = float(m.group(5))
            state.tok_per_sec = float(m.group(6))
            state.mfu = float(m.group(7))
            state.epoch = int(m.group(8))
            state.remaining = m.group(9).strip()

    return state


def parse_results_tsv(path: Path) -> list[dict[str, str]]:
    """Parse results.tsv (tab-separated, with header). Returns rows newest-first."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split("\t")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = [v.strip() for v in line.split("\t")]
        rows.append(dict(zip(header, values, strict=False)))
    return list(reversed(rows))


def parse_remote_toml(path: Path) -> RemoteConfig | None:
    """Parse remote.toml (simple key = value). Returns None if missing."""
    if not path.exists():
        return None
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"').strip("'")
    try:
        return RemoteConfig(
            host=data["host"],
            user=data["user"],
            key_path=data["key_path"],
            workspace=data.get("workspace", "/workspace/skylab"),
            num_gpus=int(data.get("num_gpus", "1")),
        )
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# GPU probing with 30s cache
# ---------------------------------------------------------------------------

_gpu_cache: tuple[float, list[GpuInfo]] | None = None
_GPU_CACHE_TTL = 30.0
_NVIDIA_SMI_FIELDS = (
    "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
)


def probe_gpu_status(config: RemoteConfig) -> list[GpuInfo]:
    """SSH to remote host and query nvidia-smi. Cached for 30s."""
    global _gpu_cache
    now = time.time()
    if _gpu_cache is not None and now - _gpu_cache[0] < _GPU_CACHE_TTL:
        return _gpu_cache[1]

    key = str(Path(config.key_path).expanduser())
    cmd = [
        "ssh",
        "-i",
        key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
        f"{config.user}@{config.host}",
        f"nvidia-smi --query-gpu={_NVIDIA_SMI_FIELDS} --format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)  # noqa: S603
        gpus: list[GpuInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    utilization=int(parts[1]),
                    memory_used=int(parts[2]),
                    memory_total=int(parts[3]),
                    temperature=int(parts[4]),
                    power=float(parts[5]),
                )
            )
        _gpu_cache = (now, gpus)
        return gpus
    except Exception:
        _gpu_cache = (now, [])
        return []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_HISTORY_PATH = WORKDIR / "monitor_history.json"


def maybe_persist_completed(run: RunState, history: list[dict[str, str]]) -> None:
    """If run is completed, append to monitor_history.json (deduplicates)."""
    if run.status != "completed" or not run.summary:
        return

    existing: list[dict[str, Any]] = []
    if _HISTORY_PATH.exists():
        try:
            existing = json.loads(_HISTORY_PATH.read_text())
        except json.JSONDecodeError:
            existing = []

    # Deduplicate: compare summary metrics against last entry
    entry: dict[str, Any] = {
        "timestamp": time.time(),
        "summary": run.summary,
        "commit": history[0].get("commit", "") if history else "",
    }
    if existing:
        last = existing[-1]
        if last.get("summary") == entry["summary"]:
            return

    existing.append(entry)
    _HISTORY_PATH.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "training": "#f0c040",
    "completed": "#4caf50",
    "idle": "#888",
    "no_data": "#888",
}

_ROW_COLOR = {
    "keep": "#4caf50",
    "discard": "#e05555",
    "crash": "#e05555",
    "training": "#f0c040",
}

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace;
       font-size: 14px; padding: 20px; }
h1 { color: #fff; font-size: 18px; letter-spacing: 2px; margin-bottom: 20px; }
h2 { color: #aaa; font-size: 13px; letter-spacing: 1px; margin-bottom: 10px;
     text-transform: uppercase; }
.panel { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
         padding: 16px; margin-bottom: 16px; }
.metric { display: inline-block; margin-right: 20px; margin-bottom: 6px; }
.metric .label { color: #888; font-size: 12px; }
.metric .value { color: #fff; font-size: 15px; font-weight: bold; }
.progress-bar-bg { background: #333; border-radius: 4px; height: 12px;
                   width: 100%; margin: 10px 0; }
.progress-bar { background: #f0c040; border-radius: 4px; height: 12px;
                transition: width 0.3s; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
                font-size: 12px; font-weight: bold; letter-spacing: 1px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { color: #888; text-align: left; padding: 6px 10px; border-bottom: 1px solid #333; }
td { padding: 6px 10px; border-bottom: 1px solid #222; }
tr:last-child td { border-bottom: none; }
.gpu-row { margin-bottom: 8px; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 10px; }
.summary-item { background: #222; border-radius: 4px; padding: 8px 12px; }
.summary-key { color: #888; font-size: 11px; }
.summary-val { color: #fff; font-size: 14px; font-weight: bold; }
.ts { color: #555; font-size: 11px; float: right; }
"""


def _badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "#888")
    return f'<span class="status-badge" style="background:{color};color:#111">{status.upper()}</span>'


def _metric(label: str, value: str) -> str:
    return (
        f'<span class="metric">'
        f'<span class="label">{label}</span><br>'
        f'<span class="value">{value}</span>'
        f"</span>"
    )


def render_html(state: DashboardState) -> str:
    run = state.run
    ts = time.strftime("%H:%M:%S", time.localtime(state.timestamp))

    # --- Current Run panel ---
    pct = run.progress_pct
    bar = (
        f'<div class="progress-bar-bg">'
        f'<div class="progress-bar" style="width:{pct:.1f}%"></div>'
        f"</div>"
    )
    tok_k = f"{run.tok_per_sec / 1000:.1f}k" if run.tok_per_sec else "—"
    run_panel = f"""
<div class="panel">
  <h2>Current Run <span class="ts">{ts}</span></h2>
  {_badge(run.status)}
  {bar}
  <div style="margin-top:8px">
    {_metric("Step", str(run.step) if run.step else "—")}
    {_metric("Progress", f"{pct:.1f}%" if pct else "—")}
    {_metric("Loss", f"{run.loss:.4f}" if run.loss else "—")}
    {_metric("MFU", f"{run.mfu:.1f}%" if run.mfu else "—")}
    {_metric("Tok/sec", tok_k)}
    {_metric("LR mult", f"{run.lr_mult:.3f}" if run.lr_mult else "—")}
    {_metric("Epoch", str(run.epoch) if run.epoch else "—")}
    {_metric("Remaining", run.remaining or "—")}
  </div>
</div>"""

    # --- GPU panel ---
    if state.gpus:
        gpu_rows = ""
        for g in state.gpus:
            vram = f"{g.memory_used / 1024:.1f}/{g.memory_total / 1024:.1f} GB"
            gpu_rows += (
                f'<div class="gpu-row">'
                f"{_metric('GPU', g.name)}"
                f"{_metric('Util', f'{g.utilization}%')}"
                f"{_metric('VRAM', vram)}"
                f"{_metric('Temp', f'{g.temperature}°C')}"
                f"{_metric('Power', f'{g.power:.0f}W')}"
                f"</div>"
            )
        gpu_panel = (
            f'<div class="panel"><h2>GPU Status (cached 30s)</h2>{gpu_rows}</div>'
        )
    else:
        gpu_panel = ""

    # --- History panel ---
    if state.history:
        rows_html = ""
        for i, row in enumerate(state.history):
            st = row.get("status", "")
            color = _ROW_COLOR.get(st, "#e0e0e0")
            num = len(state.history) - i
            rows_html += (
                f"<tr>"
                f"<td>{num}</td>"
                f"<td>{row.get('commit', '')[:7]}</td>"
                f"<td>{row.get('val_bpb', '')}</td>"
                f"<td>{row.get('memory_gb', '')}</td>"
                f'<td style="color:{color}">{st}</td>'
                f"<td>{row.get('description', '')}</td>"
                f"</tr>"
            )
        history_panel = f"""
<div class="panel">
  <h2>Experiment History</h2>
  <table>
    <tr><th>#</th><th>Commit</th><th>val_bpb</th><th>Mem (GB)</th><th>Status</th><th>Description</th></tr>
    {rows_html}
  </table>
</div>"""
    else:
        history_panel = '<div class="panel"><h2>Experiment History</h2><p style="color:#555">No results.tsv found.</p></div>'

    # --- Summary panel ---
    if run.summary:
        items = "".join(
            f'<div class="summary-item"><div class="summary-key">{k}</div>'
            f'<div class="summary-val">{v}</div></div>'
            for k, v in run.summary.items()
        )
        summary_panel = f'<div class="panel"><h2>Last Completed Run Summary</h2><div class="summary-grid">{items}</div></div>'
    else:
        summary_panel = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>Skylab Monitor</title>
<style>{_CSS}</style>
</head>
<body>
<h1>SKYLAB EXPERIMENT MONITOR</h1>
{run_panel}
{gpu_panel}
{history_panel}
{summary_panel}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(out_path: Path) -> None:
    """Generate a static report.html from monitor_history.json."""
    if not _HISTORY_PATH.exists():
        print("No monitor_history.json found — nothing to report.")
        return

    entries: list[dict[str, Any]] = json.loads(_HISTORY_PATH.read_text())
    rows_html = ""
    for e in reversed(entries):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("timestamp", 0)))
        commit = e.get("commit", "")[:7]
        summary = e.get("summary", {})
        cells = " ".join(f"<td>{v}</td>" for v in summary.values())
        rows_html += f"<tr><td>{ts}</td><td>{commit}</td>{cells}</tr>"

    if entries:
        sample = entries[-1].get("summary", {})
        headers = " ".join(f"<th>{k}</th>" for k in sample.keys())
        header_row = f"<tr><th>Timestamp</th><th>Commit</th>{headers}</tr>"
    else:
        header_row = "<tr><th>No data</th></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Skylab Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>SKYLAB EXPERIMENT REPORT</h1>
<div class="panel">
  <h2>Run History ({len(entries)} entries)</h2>
  <table>{header_row}{rows_html}</table>
</div>
</body>
</html>"""
    out_path.write_text(html)
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def _collect_state() -> DashboardState:
    run = parse_run_log(WORKDIR / "run.log")
    history = parse_results_tsv(WORKDIR / "results.tsv")
    maybe_persist_completed(run, history)

    remote_cfg = parse_remote_toml(WORKDIR / "remote.toml")
    gpus = probe_gpu_status(remote_cfg) if remote_cfg else []

    return DashboardState(run=run, gpus=gpus, history=history)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress per-request logs

    def do_GET(self) -> None:
        if self.path == "/api/status":
            state = _collect_state()
            body = json.dumps(asdict(state)).encode()
            self._respond(200, "application/json", body)
        elif self.path == "/" or self.path == "":
            state = _collect_state()
            body = render_html(state).encode()
            self._respond(200, "text/html; charset=utf-8", body)
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int) -> None:
    """Start the HTTP server and block until Ctrl+C."""
    server = HTTPServer(("localhost", port), _Handler)
    print(f"Skylab monitor running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Skylab experiment monitor")
    parser.add_argument(
        "--port", type=int, default=8080, help="HTTP port (default 8080)"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate report.html and exit"
    )
    args = parser.parse_args()

    if args.report:
        generate_report(WORKDIR / "report.html")
        return

    serve(args.port)


if __name__ == "__main__":
    main()
