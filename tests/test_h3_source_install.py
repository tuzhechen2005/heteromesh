"""Reproduce checkout conversion using real Git, retaining raw source identity."""
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('workflow',['h3-checkpoints.yml','h3-runtime.yml','weight-loader.yml'])
def test_vcs_install_preserves_source_bytes_under_autocrlf(tmp_path,workflow):
    text=(ROOT/'.github/workflows'/workflow).read_text()
    # Exercise the environment supplied to pip's git subprocess, not runtime normalization.
    env=dict(os.environ,GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='core.autocrlf',GIT_CONFIG_VALUE_0='true')
    settings=dict(re.findall(r"^\s+(GIT_CONFIG_\w+): ['\"]?([^'\"\s]+)",text,re.M))
    env.update(settings)
    source=tmp_path/'source';source.mkdir()
    def git(*args,cwd=source):return subprocess.run(['git',*args],cwd=cwd,env=env,check=True,capture_output=True)
    git('init');git('config','user.name','test');git('config','user.email','test@example.invalid')
    raw=b'first line\nsecond line\n';(source/'pinned.py').write_bytes(raw)
    git('-c','core.autocrlf=false','add','pinned.py');git('commit','-m','source')
    (source/'setup.py').write_text("from setuptools import setup\nsetup(name='h3-source-probe',version='0.0',py_modules=['pinned'])\n")
    git('-c','core.autocrlf=false','add','setup.py');git('commit','-m','package')
    wheels=tmp_path/'wheels';wheels.mkdir()
    import sys,zipfile
    result=subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','-w',str(wheels),'git+'+source.as_uri()],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    with zipfile.ZipFile(next(wheels.glob('*.whl'))) as wheel:
        assert wheel.read('pinned.py')==raw,'pip VCS install converted pinned source bytes'
    dest=tmp_path/'installed';git('clone',str(source),str(dest))
    assert (dest/'pinned.py').read_bytes()==raw,'pip VCS clone converted pinned LF source to CRLF'
