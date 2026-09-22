"""Read-only live display for an autonomous experiment; never starts model work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def age(value, now):
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        seconds = max(0, int((now - stamp).total_seconds()))
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    except (TypeError, ValueError):
        return "unknown"


def pid_status(pid):
    if os.name != "posix" or not isinstance(pid, int) or pid <= 0:
        return "not checked"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "MISSING (saved progress may be stale)"
    except PermissionError:
        return "exists (different owner)"
    return "present"


def log_tail(path, limit=6):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 8192))
            return handle.read().decode("utf-8", errors="replace").splitlines()[-limit:]
    except OSError as exc:
        return [f"Log unavailable: {exc}"]


def render(root, interval, *, height=None):
    now = datetime.now(timezone.utc)
    lines = ["IAAA autonomous experiment | LIVE PROGRESS (read-only)",
             f"Refreshed: {now.astimezone().isoformat(timespec='seconds')} | every {interval:g}s",
             f"Experiment: {root.name}", ""]
    state = None
    try:
        state = json.loads((root / "results" / "progress.json").read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("progress.json must contain an object")
    except (OSError, ValueError) as exc:
        lines += [f"Progress unavailable: {exc}", "Waiting for a readable progress.json; no evaluation is started here."]
    else:
        lines += [f"Status: {state.get('status', 'unknown')} | Stage: {state.get('stage', 'initializing')}",
                  f"Sessions: {state.get('completed', 0)} completed, {state.get('failed', 0)} failed / {state.get('total', '?')}",
                  f"Runner PID: {state.get('pid', '?')} | {pid_status(state.get('pid'))}",
                  f"Progress last changed: {age(state.get('updated_at'), now)} ago | Elapsed: {age(state.get('started_at'), now)}",
                  "", "Active sessions (last event; age is not a timeout):"]
        for key, event in sorted(state.get("active", {}).items()):
            lines += [f"  {key}",
                      f"    {event.get('detail', '?')} | {event.get('event', '?')} | {age(event.get('at'), now)} ago"]
        if not state.get("active"):
            lines.append("  No active model events recorded.")
        if state.get("error"):
            lines += ["", f"ERROR: {state.get('error_type', '')}: {state['error']}"]
        lines += ["", "Completed stages: " + ", ".join(map(str, state.get("completed_stages", [])))]
    lines += ["", "Recent runner.log:", *["  " + line for line in log_tail(root / "runner.log")], "",
              "Session counts advance after all required arms reach success or recorded failure.",
              "text = trajectory only; both = trajectory + review/image evidence.",
              "Screen: Ctrl-a then 0 = runner log; Ctrl-a then 1 = progress; Ctrl-a then d = detach.",
              "This viewer does not change predictions, requests, or the experiment protocol."]
    if height is not None and len(lines) > height and isinstance(state, dict):
        # A detached screen is often 80x24. More concurrent sessions must not
        # scroll the status/counts off screen on every refresh. Rotate pages of
        # active sessions while keeping the header and controls visible.
        header = lines[:3] + lines[4:8]
        if state.get("error"):
            header.append(f"ERROR: {state.get('error_type', '')}: {state['error']}")
        footer = ["Completed: " + ", ".join(map(str, state.get("completed_stages", []))),
                  "Latest log: " + (log_tail(root / "runner.log", limit=1) or ["(empty)"])[-1],
                  "Screen: Ctrl-a then 0=log, 1=progress, d=detach. Read-only viewer."]
        events = sorted(state.get("active", {}).items())
        capacity = max(1, height - len(header) - len(footer) - 1)
        pages = max(1, math.ceil(len(events) / capacity))
        page_size = max(1, math.ceil(len(events) / pages))
        page = int(now.timestamp() // interval) % pages
        selected = events[page * page_size:(page + 1) * page_size]
        active = [f"{key} | {event.get('detail', '?')} | {age(event.get('at'), now)} ago"
                  for key, event in selected] or ["No active model events recorded."]
        lines = header + [f"Active: {len(events)} sessions | page {page + 1}/{pages} (rotates every {interval:g}s)"] + active + footer
        lines = lines[:max(1, height)]
    # Neutralize terminal control characters in saved error/log text.
    return "\n".join("".join(c if c.isprintable() else " " for c in line) for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true", help="Print one snapshot without terminal controls")
    args = parser.parse_args()
    if not 1 <= args.interval <= 3600:
        parser.error("--interval must be between 1 and 3600 seconds")
    try:
        while True:
            terminal = shutil.get_terminal_size()
            interactive = sys.stdout.isatty() and not args.once
            output = render(args.experiment_root, args.interval,
                            height=max(1, terminal.lines - 1) if interactive else None)
            if sys.stdout.isatty() and not args.once:
                width = terminal.columns - 1
                output = "\n".join(line[:max(1, width)] for line in output.splitlines())
                print("\033[H\033[2J" + output, flush=True)
            else:
                print(output, flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Progress viewer closed; the evaluation is unaffected.", flush=True)


if __name__ == "__main__":
    main()
