import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
import pytest
from tools.development_artifacts import ArtifactError, inspect_distribution, write_manifest

def test_manifest_is_commit_bound_and_explicitly_development(tmp_path):
    wheel=tmp_path/'heteromesh-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as archive:archive.writestr('heteromesh/cli.py','pass')
    report=inspect_distribution(wheel)
    output=write_manifest(tmp_path,[report],commit_sha='a'*40,platform='test')
    data=json.loads(output.read_text(encoding='utf-8'))
    assert data['commit_sha']=='a'*40
    assert data['delivery_kind']=='development'
    assert data['hardware_validation']=='NOT_RUN'
    assert data['artifacts'][0]['sha256']==hashlib.sha256(wheel.read_bytes()).hexdigest()
    with pytest.raises(ArtifactError):write_manifest(tmp_path,[report],commit_sha='main',platform='test')

@pytest.mark.parametrize('name',['../outside','/absolute','heteromesh/identity/admin.token','heteromesh/.git/config','heteromesh/secret.pem','heteromesh/weights.safetensors'])
def test_forbidden_archive_contents_rejected(tmp_path,name):
    wheel=tmp_path/'bad.whl'
    with zipfile.ZipFile(wheel,'w') as archive:archive.writestr(name,'sensitive')
    with pytest.raises(ArtifactError):inspect_distribution(wheel)

def test_source_distribution_symlink_rejected(tmp_path):
    source=tmp_path/'source.tar.gz'
    with tarfile.open(source,'w:gz') as archive:
        link=tarfile.TarInfo('package/link');link.type=tarfile.SYMTYPE;link.linkname='/private'
        archive.addfile(link)
    with pytest.raises(ArtifactError):inspect_distribution(source)

def test_source_archive_uses_committed_bytes_not_dirty_checkout(tmp_path):
    import subprocess
    from tools.development_artifacts import archive_source
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    (repo/'module.py').write_text('committed = True\n',encoding='utf-8')
    subprocess.run(['git','add','module.py'],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Artifact Test','-c','user.email=artifact@example.invalid','commit','-m','fixture'],cwd=repo,check=True,capture_output=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    (repo/'module.py').write_text('dirty = True\n',encoding='utf-8')
    (repo/'untracked.py').write_text('secret = True\n',encoding='utf-8')
    target=tmp_path/'source'
    archive_source(repo,sha,target)
    assert (target/'module.py').read_text(encoding='utf-8')=='committed = True\n'
    assert not (target/'untracked.py').exists()
    with pytest.raises(ArtifactError):archive_source(repo,'0'*40,tmp_path/'bad')
