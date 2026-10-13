"""E1 reception sentinels: tiny bodies, real Worker/PinnedClient control flow."""
import hashlib
import io
import struct
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
from diffusers import MiniMaxH3Scheduler
from heteromesh.h3_network import StageExecutor, SCHEDULER, tensor_to_frame
from heteromesh.client import PinnedClient
from heteromesh.worker import Worker


class Response:
    status=200
    def __init__(self,raw,length):
        self.stream=io.BytesIO(raw);self.length=length;self.requests=[];self.extra_headers=[]
        self.payload_start=4+struct.unpack('>I',raw[:4])[0];self.payload_read=0
    def getheaders(self):return ([] if self.length is None else [('Content-Length',self.length)])+self.extra_headers
    def getheader(self,name,default=None):return self.length if name.lower()=='content-length' else default
    def read(self,count):
        self.requests.append(count);before=self.stream.tell();result=self.stream.read(count)
        self.payload_read+=max(0,self.stream.tell()-max(before,self.payload_start))
        return result


class Connection:
    def __init__(self,response):self.response=response;self.closed=False
    def request(self,*args,**kwargs):pass
    def getresponse(self):return self.response
    def close(self):self.closed=True


class Client(PinnedClient):
    def __init__(self,task,responses):
        super().__init__('fixture',0,'a'*64,timeout=1)
        self.task=task;self.responses=iter(responses);self.connections=[];self.events=[]
    def _connect(self):
        c=Connection(next(self.responses));self.connections.append(c);return c
    def request(self,method,path,body=None,**kwargs):
        self.events.append(path)
        return {'task':self.task} if path.endswith('/lease') else {}


def setup_case(payload_allowance=12):
    pair=[MiniMaxH3Scheduler(),MiniMaxH3Scheduler()]
    for s in pair:s.set_timesteps(sigmas=[1.,.5,0.])
    stage=StageExecutor('a'*64,'scheduler','scheduler_pair',schedulers=pair,
        max_boundary_bytes=3*(len(SCHEDULER)*65540+payload_allowance))
    raws=[tensor_to_frame(n,torch.zeros(1,2,1)) for n in SCHEDULER]
    task={'operation':'h3_scheduler_pair_v1','manifest_digest':'a'*64,
        'parameters':{'profile_digest':'a'*64,'stage_id':'scheduler'},'step_index':0,
        'inputs':dict(zip(SCHEDULER,[hashlib.sha256(r).hexdigest() for r in raws])),
        'outputs':{'hidden_states':{'dtype':'float32','shape':[1,2,1]},'audio_hidden_states':{'dtype':'float32','shape':[1,2,1]}},
        'job_id':'fixture','recovery_epoch':0,'attempt_id':'fixture'}
    return stage,raws,task


def test_declared_oversize_reads_zero_body():
    stage,raws,task=setup_case()
    responses=[Response(r,str(65540+13)) for r in raws]
    client=Client(task,responses)
    assert not Worker(client,{}, {task['operation']:stage}).run_once()
    assert responses[0].requests==[]
    assert len(client.connections)==1 and client.connections[0].closed


def test_cumulative_budget_does_not_read_crossing_payload():
    stage,raws,task=setup_case()
    responses=[Response(r,str(len(r))) for r in raws];client=Client(task,responses)
    assert not Worker(client,{}, {task['operation']:stage}).run_once()
    assert responses[0].payload_read==8
    assert responses[1].payload_read==0
    assert len(client.connections)==2 and all(c.closed for c in client.connections)


@pytest.mark.parametrize('kind',['missing','invalid','lied','header_length','header_payload'])
def test_untrusted_lengths_are_rejected_with_bounded_reads(kind):
    stage,raws,task=setup_case()
    raw=raws[0];length=str(len(raw))
    if kind=='missing':length=None
    if kind=='invalid':length='not-a-length'
    if kind=='lied':length=str(len(raw)-1)
    if kind=='header_length':raw=struct.pack('>I',65537)+raw[4:]
    if kind=='header_payload':
        raw=tensor_to_frame(SCHEDULER[0],torch.zeros(1,4,1));length=str(len(raw))
        task['inputs'][SCHEDULER[0]]=hashlib.sha256(raw).hexdigest()
    responses=[Response(raw,length)]+[Response(r,str(len(r))) for r in raws[1:]]
    client=Client(task,responses)
    assert not Worker(client,{}, {task['operation']:stage}).run_once()
    assert responses[0].payload_read==0
    assert all(0<n<=65536 for n in responses[0].requests)
    assert len(client.connections)==1 and client.connections[0].closed


@pytest.mark.parametrize('kind',['duplicate','comma','negative','plus','transfer','cl_transfer'])
def test_http_framing_ambiguity_reads_zero_body(kind):
    stage,raws,task=setup_case();r=Response(raws[0],str(len(raws[0])))
    if kind=='duplicate':r.extra_headers=[('Content-Length',r.length)]
    if kind=='comma':r.length+=','+r.length
    if kind=='negative':r.length='-1'
    if kind=='plus':r.length='+'+r.length
    if kind in ('transfer','cl_transfer'):
        r.extra_headers=[('Transfer-Encoding','chunked')]
        if kind=='transfer':r.length=None
    client=Client(task,[r]+[Response(x,str(len(x))) for x in raws[1:]])
    assert not Worker(client,{}, {task['operation']:stage}).run_once()
    assert r.requests==[] and client.connections[0].closed
