import copy,json
from pathlib import Path
import pytest
from heteromesh.capacity import validate_placement,CapacityError
ROOT=Path(__file__).resolve().parents[1]

def fixture():
    return json.loads((ROOT/'fixtures/capabilities.json').read_text()),json.loads((ROOT/'fixtures/manifest.json').read_text())['fragments']

def test_sums_resident_but_reuses_sequential_workspaces_and_accounts_loading():
    cap,fs=fixture(); other=copy.deepcopy(fs[0]);other['id']='block1';fs.append(other)
    report=validate_placement(cap,fs)
    assert report['host_peak_bytes']==270 # other resident + one fragment loading
    assert report['accelerator_peak_bytes']==0
    cap['memory']['host_budget_bytes']=269
    with pytest.raises(CapacityError):validate_placement(cap,fs)

def test_unified_merge_and_separate_gpu_budgets():
    cap,fs=fixture();fs[0]['memory']['accelerator']=copy.deepcopy(fs[0]['memory']['host'])
    assert validate_placement(cap,fs)['host_peak_bytes']==340
    cap['memory'].update(unified=False,host_budget_bytes=400,accelerator_budget_bytes=169)
    with pytest.raises(CapacityError):validate_placement(cap,fs)
    cap['memory']['accelerator_budget_bytes']=170
    assert validate_placement(cap,fs)['accelerator_peak_bytes']==170

@pytest.mark.parametrize('change',['unknown','operator','dtype','backend','wire','bool','negative','inadequate_loading'])
def test_plan_rejects_unproven_capacity_and_capabilities(change):
    cap,fs=fixture()
    if change=='unknown':cap['memory']['host_budget_bytes']=None
    if change=='operator':cap['supported_ops']=[]
    if change=='dtype':cap['compute_dtypes']=[]
    if change=='backend':cap['backend']='metal'
    if change=='wire':cap['wire_dtypes']=[]
    if change=='bool':fs[0]['memory']['host']['workspace_bytes']=True
    if change=='negative':fs[0]['memory']['host']['workspace_bytes']=-1
    if change=='inadequate_loading':fs[0]['memory']['host']['loading_peak_bytes']=90
    with pytest.raises(CapacityError):validate_placement(cap,fs)
