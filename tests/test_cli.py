import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from heteromesh.cli import main, private_json, prepared_validator
from heteromesh.service import Coordinator


def test_secret_file_is_exclusive_and_not_overwritten(tmp_path):
    p=tmp_path/'secret.json'
    private_json(p,{'token':'local-test-value'})
    with pytest.raises(FileExistsError):private_json(p,{'token':'different'})
    assert json.loads(p.read_text())['token']=='local-test-value'
    if os.name!='nt':assert p.stat().st_mode & 0o077==0


def test_cli_runs_dependent_job_in_two_actual_worker_processes(tmp_path, capsys):
    root=tmp_path/'server'
    server=Coordinator(root,validate_job=prepared_validator(root))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    common=['--state-dir',str(root),'--host','127.0.0.1','--port',str(server.port)]
    configs=[];node_ids=[]
    try:
        for i in range(2):
            pairing=tmp_path/f'pair-{i}.json';config=tmp_path/f'node-{i}.json'
            assert main(['pair',*common,'--out',str(pairing)])==0
            assert main(['join','--pairing-file',str(pairing),'--config',str(config),'--backend','numpy'])==0
            configs.append(config);node_ids.append(json.loads(config.read_text())['node_id'])
        output=capsys.readouterr().out
        assert 'token' not in output
        assert main(['submit-tiny',*common,'--nodes',*node_ids,'--steps','2'])==0
        job_id=json.loads(capsys.readouterr().out)['job_id']
        env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()))
        for config in configs*2:
            run=subprocess.run([sys.executable,'-m','heteromesh','worker','--config',str(config),'--once'],env=env,capture_output=True,text=True,timeout=30)
            assert run.returncode==0,run.stderr
        assert main(['status',*common,'--job',job_id])==0
        assert json.loads(capsys.readouterr().out)['state']=='succeeded'
        assert main(['verify-tiny',*common,'--job',job_id])==0
        report=json.loads(capsys.readouterr().out)
        assert report['numerical_match'] is True
        assert report['evidence']=='local-process-or-user-connected-nodes'
    finally:
        server.shutdown();thread.join();server.close()
