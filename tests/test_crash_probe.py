"""Small real subprocess tests; no Torch or coordinator dependency."""
import pytest
from _crash_probe import terminate_at_marker


def test_early_exit_reports_actual_code_and_output(tmp_path):
    with pytest.raises(AssertionError) as caught:
        terminate_at_marker("import sys; print('synthetic import phase',flush=True); sys.exit(7)",[],tmp_path/'marker',timeout=2)
    message=str(caught.value)
    assert 'child exited before marker' in message and 'exit_code_before_stop=7' in message
    assert 'synthetic import phase' in message


def test_deadline_reports_phase_elapsed_and_stack(tmp_path):
    with pytest.raises(AssertionError) as caught:
        terminate_at_marker("import time; print('synthetic blocked phase',flush=True); time.sleep(10)",[],tmp_path/'marker',timeout=.5)
    message=str(caught.value)
    assert 'marker deadline exceeded' in message and 'elapsed=' in message
    assert 'exit_code_before_stop=None' in message and 'synthetic blocked phase' in message
    assert 'Timeout (' in message and 'line' in message


def test_marker_is_actually_terminated_without_pipe_backpressure(tmp_path):
    marker=tmp_path/'marker'
    code="import time; print('x'*100000,flush=True); Path(sys.argv[1]).write_text('ready'); time.sleep(10)"
    report=terminate_at_marker(code,[marker],marker,timeout=2)
    assert marker.exists() and 'exit_code_after_stop=0\n' not in report
    assert 'x'*100000 in report
