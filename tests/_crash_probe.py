"""Bounded real termination with startup phases and genuine exit diagnostics."""
import subprocess
import sys
import time
import tempfile
import os
import signal

PRELUDE='''import faulthandler,hashlib,json,time,sys
from pathlib import Path
_probe_started=time.monotonic()
faulthandler.enable()
faulthandler.dump_traceback_later(STACK_AFTER)
def probe_stage(stage):
 print(json.dumps({'stage':stage,'elapsed_seconds':round(time.monotonic()-_probe_started,6)}),flush=True)
def probe_module(name):
 path=Path(sys.modules[name].__file__)
 print(json.dumps({'module':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),flush=True)
probe_stage('python_started')
'''


def terminate_at_marker(code,args,marker,*,timeout):
    assert not marker.exists(),'stale marker before launch'
    started=time.monotonic()
    # File capture cannot fill an undrained PIPE while faulthandler writes stacks.
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        prelude=PRELUDE.replace('STACK_AFTER',repr(min(10,timeout/2)))
        p=subprocess.Popen([sys.executable,'-c',prelude+code,*map(str,args)],stdout=out,stderr=err)
        try:
            deadline=started+timeout
            while not marker.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.02)
            before=p.poll();reached=marker.exists()
            if reached:p.terminate()
            elif before is None:p.kill()
            p.wait(timeout=10)
        finally:
            if p.poll() is None:p.kill();p.wait(timeout=10)
        out.seek(0);err.seek(0)
        report=f'elapsed={time.monotonic()-started:.3f}s exit_code_before_stop={before} exit_code_after_stop={p.returncode}\nstdout:\n{out.read().decode(errors="replace")}\nstderr:\n{err.read().decode(errors="replace")}'
        if not reached:
            reason='marker deadline exceeded' if before is None else 'child exited before marker'
            raise AssertionError(reason+' '+report)
        assert before is None,'child exited before termination: '+report
        expected=-signal.SIGTERM if os.name=='posix' else 1
        assert p.returncode==expected,'child exited without expected termination status: '+report
        return report
