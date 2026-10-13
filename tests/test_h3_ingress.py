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


@pytest.mark.parametrize('kind',['profile','stage','inputs_missing','inputs_extra','step','no_policy'])
def test_local_preflight_rejects_before_first_artifact_request(kind):
    stage,raws,task=setup_case(32)
    operation=stage
    if kind=='profile':task['manifest_digest']='b'*64
    if kind=='stage':task['parameters']['stage_id']='uninstalled'
    if kind=='inputs_missing':del task['inputs'][SCHEDULER[0]]
    if kind=='inputs_extra':task['inputs']['unexpected']='b'*64
    if kind=='step':task['step_index']=True
    if kind=='no_policy':operation=lambda task,frames:stage(task,frames)
    task['max_boundary_bytes']=2**40  # A remote field never authorizes more reception.
    client=Client(task,[Response(r,str(len(r))) for r in raws])
    assert not Worker(client,{}, {task['operation']:operation}).run_once()
    assert client.connections==[]
    assert client.events[-1]=='/v1/work/error'


def test_actual_encoded_frame_limit_retains_prior_inputs():
    from heteromesh.worker import InputPolicy
    stage,raws,task=setup_case(32)
    # This local policy isolates encoded-frame accounting from the conservative
    # H3 header reservation. The second frame exceeds total by one byte.
    class Installed:
        def input_policy(self,task):
            return InputPolicy(len(raws[0])+len(raws[1])-1,32,tuple((n,('float32',)) for n in SCHEDULER))
        def __call__(self,task,frames):raise AssertionError('must not execute')
    responses=[Response(r,str(len(r))) for r in raws];client=Client(task,responses)
    assert not Worker(client,{}, {task['operation']:Installed()}).run_once()
    assert responses[0].payload_read==8
    assert responses[1].requests==[]
    assert len(client.connections)==2


@pytest.mark.parametrize('kind',['valid','frame_over_one','payload_over_one','name','dtype','payload_hash','artifact_hash','truncated','trailing'])
def test_bounded_client_exact_limits_and_raw_header_identity(kind):
    import json
    from heteromesh.protocol import decode_tensor
    _,raws,task=setup_case(32)
    frame=decode_tensor(raws[0])
    header=json.dumps(frame.header,indent=2).encode()  # D-07: legal noncanonical wire header.
    raw=struct.pack('>I',len(header))+header+frame.payload
    digest=hashlib.sha256(raw).hexdigest();max_frame=len(raw);max_payload=8
    name=SCHEDULER[0];dtypes=('float32',);length=str(len(raw))
    if kind=='frame_over_one':max_frame-=1
    if kind=='payload_over_one':max_payload-=1
    if kind=='name':name='wrong'
    if kind=='dtype':dtypes=('bfloat16',)
    if kind=='payload_hash':raw=raw[:-1]+bytes([raw[-1]^1]);digest=hashlib.sha256(raw).hexdigest()
    if kind=='artifact_hash':digest='b'*64
    if kind=='truncated':raw=raw[:-1]
    if kind=='trailing':raw+=b'x'
    r=Response(raw,length);client=Client(task,[r])
    options=dict(max_frame_bytes=max_frame,max_payload_bytes=max_payload,expected_name=name,allowed_dtypes=dtypes)
    if kind=='valid':assert client.get_artifact(digest,task,**options)==raw
    else:
        with pytest.raises(ValueError):client.get_artifact(digest,task,**options)
    assert client.connections[0].closed
    assert all(0<n<=4*1024*1024 for n in r.requests)
    if kind=='frame_over_one':assert r.requests==[]
    if kind in ('payload_over_one','name','dtype'):assert r.payload_read==0


@pytest.mark.parametrize('status',[401,403])
def test_bounded_artifact_errors_keep_remote_error_and_small_json_limit(status):
    from heteromesh.client import RemoteError
    import json
    _,raws,task=setup_case(32)
    r=Response(raws[0],None);r.status=status
    r.stream=io.BytesIO(json.dumps({'error':{'code':'UNAUTHORIZED'}}).encode())
    client=Client(task,[r])
    with pytest.raises(RemoteError) as caught:
        client.get_artifact('a'*64,task,max_frame_bytes=16,max_payload_bytes=8)
    assert caught.value.status==status and caught.value.body['error']['code']=='UNAUTHORIZED'
    assert r.requests==[65537] and client.connections[0].closed


@pytest.mark.parametrize('allowance',[0,12])
def test_real_pinned_https_worker_rejects_before_crossing_payload(tmp_path,allowance):
    import threading
    from heteromesh.service import Coordinator
    stage,raws,_=setup_case(allowance)
    server=Coordinator(tmp_path,validate_job=lambda request,nodes:None)
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    observed=[]
    class RecordingClient(PinnedClient):
        receiving=False
        def get_artifact(self,*args,**kwargs):
            self.receiving=True
            try:return super().get_artifact(*args,**kwargs)
            finally:self.receiving=False
        def _connect(self):
            c=super()._connect()
            if self.receiving:
                getresponse=c.getresponse
                def response():
                    r=getresponse();read=r.read;counts=[];observed.append(counts)
                    def record(n):counts.append(n);return read(n)
                    r.read=record
                    return r
                c.getresponse=response
            return c
    def client(token,cls=PinnedClient):return cls('127.0.0.1',server.port,server.identity['fingerprint'],token=token)
    try:
        admin=client(server.identity['admin_token'])
        pairing=admin.request('POST','/v1/pairing',{})
        cap={'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu',
            'supported_ops':['h3_scheduler_pair_v1'],'wire_dtypes':['float32'],
            'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':16*1024*1024,'accelerator_budget_bytes':None}}
        node=client(pairing['token']).request('POST','/v1/nodes/register',{'capabilities':cap})
        inputs={}
        for name,raw in zip(SCHEDULER,raws):
            digest=hashlib.sha256(raw).hexdigest();admin.put_artifact(digest,raw);inputs[name]=digest
        task={'task_id':'scheduler','fragment_id':'scheduler','step_index':0,'node_id':node['node_id'],
            'operation':'h3_scheduler_pair_v1','parameters':{'profile_digest':'a'*64,'stage_id':'scheduler'},
            'inputs':inputs,'outputs':{'hidden_states':{'dtype':'float32','shape':[1,2,1]},'audio_hidden_states':{'dtype':'float32','shape':[1,2,1]}}}
        job=admin.request('POST','/v1/jobs',{'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[task]},headers={'Idempotency-Key':'ingress-fixture'})
        assert not Worker(client(node['token'],RecordingClient),cap,{'h3_scheduler_pair_v1':stage}).run_once()
        assert admin.request('GET','/v1/jobs/'+job['job_id'])['state']=='failed'
        assert len(observed)==(1 if allowance==0 else 2)
        # Exactly prefix+header reads on the rejecting response, no payload read.
        assert len(observed[-1])==2 and observed[-1][0]==4
        assert observed[-1][1]<=65536
        if allowance==12:assert observed[0][-2:]==[8,1]
    finally:
        server.shutdown();serving.join();server.close()
