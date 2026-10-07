import copy
import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from heteromesh.manifests import validate_manifest,validate_capabilities,ManifestError
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name',['manifest','capabilities'])
def test_contract_fixture(name):
    data=json.loads((ROOT/f'fixtures/{name}.json').read_text(encoding='utf-8'))
    Draft202012Validator(json.loads((ROOT/f'schemas/{name}.json').read_text(encoding='utf-8'))).validate(data)
    assert (validate_manifest if name=='manifest' else validate_capabilities)(data)==data

def test_manifest_refuses_unknown_code_missing_license_duplicate_ids_and_names():
    good=json.loads((ROOT/'fixtures/manifest.json').read_text(encoding='utf-8'))
    cases=[]
    x=copy.deepcopy(good);x['code']='print(1)';cases.append(x)
    x=copy.deepcopy(good);del x['license'];cases.append(x)
    x=copy.deepcopy(good);x['fragments']*=2;cases.append(x)
    x=copy.deepcopy(good);x['fragments'][0]['inputs']*=2;cases.append(x)
    x=copy.deepcopy(good);x['fragments'][0]['weights'][0]['bytes']=True;cases.append(x)
    for x in cases:
        with pytest.raises(ManifestError):validate_manifest(x)

def test_unified_must_not_double_count_and_unknown_fields_refused():
    data=json.loads((ROOT/'fixtures/capabilities.json').read_text(encoding='utf-8'))
    data['memory']['accelerator_budget_bytes']=300
    with pytest.raises(ManifestError):validate_capabilities(data)
    data['memory']['accelerator_budget_bytes']=None;data['wire_dtypes'].append('pickle')
    with pytest.raises(ManifestError):validate_capabilities(data)
