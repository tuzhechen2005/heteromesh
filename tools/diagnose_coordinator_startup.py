"""Short hosted-runner startup probe; no weights, tokens or host names in output."""
import faulthandler
import json
import time


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(10, repeat=True)
    started=time.monotonic()
    def phase(name):
        print(json.dumps({'stage':name,'elapsed_seconds':round(time.monotonic()-started,6)}),flush=True)
    phase('python_started')
    import socket
    import tempfile
    import hashlib
    from pathlib import Path
    from heteromesh import service
    phase('service_imported')
    print(json.dumps({'module':'heteromesh.service','source_sha256':hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest()}),flush=True)
    def measured(name,original):
        def call(*args,**kwargs):
            phase(name+'_begin')
            value=original(*args,**kwargs)
            phase(name+'_end')
            return value
        return call
    # Measurement only: call the original functions with identical arguments.
    socket.getfqdn=measured('getfqdn',socket.getfqdn)
    for name in ('create_identity','StateStore','SecurityStore','_CoordinatorHTTPServer'):
        setattr(service,name,measured(name,getattr(service,name)))
    with tempfile.TemporaryDirectory(prefix='heteromesh-startup-') as root:
        for mode in ('fresh','reopen'):
            phase(mode+'_coordinator_begin')
            app=service.Coordinator(root)
            phase(mode+'_coordinator_ready')
            app.close()
            phase(mode+'_coordinator_closed')
    faulthandler.cancel_dump_traceback_later()
    phase('complete')


if __name__=='__main__':main()
