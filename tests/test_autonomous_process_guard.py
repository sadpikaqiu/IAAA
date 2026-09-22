from pathlib import Path

import pytest

from scripts.autonomous_process_guard import evaluator_state


def process(tmp_path, state="S", cmd=b"python\0scripts/evaluate_autonomous.py\0--output-dir\0/old/results\0", start=99):
    directory = tmp_path / "73"
    directory.mkdir()
    (directory / "stat").write_text("73 (python worker)) " + state + " " + "0 " * 18 + str(start))
    (directory / "cmdline").write_bytes(cmd)
    return directory


def inspect(tmp_path, **kwargs):
    return evaluator_state(73, ["scripts/evaluate_autonomous.py", "/old/results"], proc_root=tmp_path, **kwargs)


def test_matching_live_process_and_pid_reuse(tmp_path):
    process(tmp_path)
    assert inspect(tmp_path)["state"] == "running"
    assert inspect(tmp_path, start_ticks=98)["state"] == "mismatch"


def test_live_wrong_command_remains_an_error(tmp_path):
    process(tmp_path, cmd=b"python\0different_job.py\0")
    assert inspect(tmp_path)["state"] == "mismatch"


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_empty_cmdline_for_terminated_process_is_normal_exit(tmp_path, state):
    process(tmp_path, state=state, cmd=b"")
    assert inspect(tmp_path)["state"] == "exited"


def test_live_empty_cmdline_must_wait_not_start_a_duplicate(tmp_path):
    process(tmp_path, cmd=b"")
    assert inspect(tmp_path)["state"] == "transition"


def test_disappearance_between_stat_and_cmdline_is_normal_exit(tmp_path):
    directory = process(tmp_path)
    (directory / "cmdline").unlink()
    assert inspect(tmp_path)["state"] == "exited"


def test_exit_during_command_read_is_normal_exit(tmp_path, monkeypatch):
    directory = process(tmp_path)
    original = Path.read_bytes
    def read(path):
        (directory / "stat").write_text("73 (python) Z " + "0 " * 18 + "99")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    assert inspect(tmp_path)["state"] == "exited"


def test_nonexistent_pid_is_exited(tmp_path):
    assert inspect(tmp_path)["state"] == "exited"
