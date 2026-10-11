"""Build and smoke-test development packages; never publishes a release."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

class ArtifactError(ValueError): pass


def _safe_name(name):
    path=PurePosixPath(name.replace('\\','/'))
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ArtifactError('Unsafe archive path')
    if any(p.lower() in {'.git','.venv','identity','artifacts','quarantine','.env'} for p in path.parts):
        raise ArtifactError('Runtime/private directory in distribution')
    if path.suffix.lower() in {'.pem','.key','.token','.safetensors','.pt','.pth'}:
        raise ArtifactError('Credential or model weight file in distribution')


def inspect_distribution(path):
    path=Path(path)
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                _safe_name(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16): raise ArtifactError('Archive symlink forbidden')
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path,'r:gz') as archive:
            for info in archive.getmembers():
                _safe_name(info.name)
                if not (info.isfile() or info.isdir()): raise ArtifactError('Archive links/devices forbidden')
    else: raise ArtifactError('Only wheels and source distributions are accepted')
    digest=hashlib.sha256()
    with path.open('rb') as source:
        while block:=source.read(1024*1024): digest.update(block)
    return {'filename':path.name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()}


def write_manifest(directory,artifacts,*,commit_sha,platform):
    if not re.fullmatch('[0-9a-f]{40}',commit_sha): raise ArtifactError('Full commit SHA required')
    result={'delivery_kind':'development','package_version':'0.1.0','commit_sha':commit_sha,
            'build_platform':platform,'hardware_validation':'NOT_RUN',
            'capability_claims':[],'artifacts':artifacts}
    path=Path(directory)/'development-manifest.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return path


def archive_source(repo,commit_sha,destination):
    if not re.fullmatch('[0-9a-f]{40}',commit_sha): raise ArtifactError('Full commit SHA required')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if actual!=commit_sha: raise ArtifactError('Requested commit differs from checked-out HEAD')
    destination=Path(destination);destination.mkdir(parents=True)
    with tempfile.TemporaryFile() as source:
        subprocess.run(['git','archive','--format=tar',commit_sha],cwd=repo,stdout=source,check=True)
        source.seek(0)
        with tarfile.open(fileobj=source,mode='r:') as archive:
            for member in archive.getmembers():
                _safe_name(member.name)
                if not (member.isfile() or member.isdir()): raise ArtifactError('Source archive links/devices forbidden')
            archive.extractall(destination,filter='data')
    return destination


def _run(args,*,cwd,env):
    subprocess.run([str(x) for x in args],cwd=cwd,env=env,check=True)


def _smoke(python,console,*,cwd,env,venv_root):
    script='''import importlib.resources as r, json, pathlib, sys
from jsonschema import Draft202012Validator
import heteromesh.cli as cli
module_path=pathlib.Path(cli.__file__).resolve()
root=pathlib.Path(sys.argv[1]).resolve()
assert module_path.is_relative_to(root), "Imported code outside clean venv"
for name in ["capabilities", "manifest", "tensor-header"]:
    schema=json.loads(r.files("heteromesh.schemas").joinpath(name+".json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
print("Installed module and packaged schemas verified")
'''
    _run([python,'-I','-c',script,venv_root],cwd=cwd,env=env)
    _run([python,'-I','-m','heteromesh','--help'],cwd=cwd,env=env)
    _run([console,'--help'],cwd=cwd,env=env)


def build_and_verify(repo,output,*,commit_sha,platform):
    if not re.fullmatch('[0-9a-f]{40}',commit_sha): raise ArtifactError('Full commit SHA required')
    repo=Path(repo).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if list(output.iterdir()): raise ArtifactError('Output directory must be empty')
    env=dict(os.environ)
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    with tempfile.TemporaryDirectory(prefix='heteromesh-source-') as source_temp:
        source=archive_source(repo,commit_sha,Path(source_temp)/'source')
        _run([sys.executable,'-m','build','--outdir',output],cwd=source,env=env)
    wheels=list(output.glob('*.whl'));sources=list(output.glob('*.tar.gz'))
    if len(wheels)!=1 or len(sources)!=1: raise ArtifactError('Expected exactly one wheel and one sdist')
    reports=[inspect_distribution(p) for p in wheels+sources]
    with tempfile.TemporaryDirectory(prefix='heteromesh-install-') as temp:
        temp=Path(temp).resolve(); venv=temp/'venv'; outside=temp/'outside'; outside.mkdir()
        _run([sys.executable,'-m','venv',venv],cwd=outside,env=env)
        binary=venv/('Scripts' if os.name=='nt' else 'bin')
        python=binary/('python.exe' if os.name=='nt' else 'python')
        console=binary/('heteromesh.exe' if os.name=='nt' else 'heteromesh')
        _run([python,'-m','pip','install',wheels[0]],cwd=outside,env=env)
        _smoke(python,console,cwd=outside,env=env,venv_root=venv)
        rebuilt=temp/'rebuilt';rebuilt.mkdir()
        _run([python,'-m','pip','wheel','--no-deps','--wheel-dir',rebuilt,sources[0]],cwd=outside,env=env)
        rebuilt_wheels=list(rebuilt.glob('*.whl'))
        if len(rebuilt_wheels)!=1: raise ArtifactError('Source distribution did not build a wheel')
        inspect_distribution(rebuilt_wheels[0])
        _run([python,'-m','pip','install','--no-deps','--force-reinstall',rebuilt_wheels[0]],cwd=outside,env=env)
        _smoke(python,console,cwd=outside,env=env,venv_root=venv)
    return write_manifest(output,reports,commit_sha=commit_sha,platform=platform)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--platform',default=sys.platform)
    args=parser.parse_args()
    print(build_and_verify(Path(__file__).resolve().parents[1],args.output_dir,commit_sha=args.commit,platform=args.platform))

if __name__=='__main__':main()
