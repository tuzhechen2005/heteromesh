import copy
import time
import pytest
from heteromesh.demo import build_plan, validate_plan, DemoError


def nodes():
    cap={'protocol_version':1,'platform':'macos','runtime':'numpy','backend':'cpu',
         'supported_ops':['tiny_transformer_block_v1'],'wire_dtypes':['float32'],
         'compute_dtypes':['float32'],'memory':{'unified':True,'host_budget_bytes':2**30,'accelerator_budget_bytes':None}}
    return [{'node_id':n,'capabilities':copy.deepcopy(cap),'last_seen':int(time.time()),'revoked':False} for n in ['node-a','node-b']]


def test_two_nodes_are_required_and_tasks_really_depend_on_previous_output():
    ns=nodes()
    with pytest.raises(DemoError): build_plan(ns[:1],steps=2)
    with pytest.raises(DemoError): build_plan([ns[0],ns[0]],steps=2)
    request,artifacts,manifest=build_plan(ns,steps=2)
    assert len(request['tasks'])==4
    assert request['tasks'][1]['inputs']['hidden']=={'task_id':'step-0-block-0','output':'hidden'}
    assert request['tasks'][2]['inputs']['hidden']=={'task_id':'step-0-block-1','output':'hidden'}
    assert len({t['node_id'] for t in request['tasks']})==2
    assert all(len(v)>4 for v in artifacts.values())
    validate_plan(request,ns)


@pytest.mark.parametrize('change',['memory','operator','stale','revoked','unknown','mutated_task','manifest','extra_task'])
def test_prepare_rejects_unsupported_or_tampered_plan(change):
    ns=nodes();request,_,_=build_plan(ns,steps=2)
    if change=='memory':ns[0]['capabilities']['memory']['host_budget_bytes']=1
    if change=='operator':ns[0]['capabilities']['supported_ops']=[]
    if change=='stale':ns[0]['last_seen']=0
    if change=='revoked':ns[0]['revoked']=True
    if change=='unknown':ns.pop()
    if change=='mutated_task':request['tasks'][1]['node_id']='node-a'
    if change=='manifest':request['manifest_digest']='0'*64
    if change=='extra_task':request['tasks'].append(copy.deepcopy(request['tasks'][-1]))
    with pytest.raises(DemoError):validate_plan(request,ns)
