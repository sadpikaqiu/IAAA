"""Read Linux evaluator liveness without confusing process exit with PID reuse.

This is an operations helper, not part of the prediction or scoring protocol.
The caller must still acquire the experiment's OS lock before inheriting files.
"""
from pathlib import Path


def evaluator_state(pid, expected_arguments, *, proc_root=Path("/proc"), start_ticks=None):
    directory = Path(proc_root) / str(pid)

    def stat():
        # The comm field is parenthesized and may itself contain spaces or ')'.
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])  # state (3), starttime (22)

    def result(state, reason):
        return {"pid": pid, "state": state, "reason": reason}

    try:
        before, generation = stat()
        if before in {"Z", "X", "x"}:
            return result("exited", "terminated_process_not_yet_reaped")
        if start_ticks is not None and generation != start_ticks:
            return result("mismatch", "pid_reused")
        arguments = (directory / "cmdline").read_bytes().split(b"\0")
        after, current_generation = stat()
    except (FileNotFoundError, ProcessLookupError):
        return result("exited", "proc_entry_disappeared")
    if generation != current_generation:
        return result("mismatch", "pid_reused_during_read")
    if after in {"Z", "X", "x"}:
        return result("exited", "terminated_during_read")
    if not any(arguments):
        # Linux may briefly expose an empty cmdline while the task is exiting.
        # Wait for disappearance/zombie state; never launch based on empty alone.
        return result("transition", "live_empty_cmdline")
    if not all(str(arg).encode() in arguments for arg in expected_arguments):
        return result("mismatch", "live_command_does_not_match")
    return {**result("running", "matching_live_process"), "start_ticks": generation}
