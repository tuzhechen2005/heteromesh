import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from heteromesh.client import PinnedClient
from heteromesh.service import Coordinator
from heteromesh.worker import Worker
from heteromesh.protocol import encode_tensor, decode_tensor

class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.server=Coordinator(Path(self.tmp.name),validate_job=lambda request,nodes:None)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.admin=self.client(self.server.identity['admin_token'])
        pair=self.admin.request('POST','/v1/pairing',{})
        self.node=self.client(pair['token']).request('POST','/v1/nodes/register',{'capabilities':{'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}}})
    def client(self,token): return PinnedClient('127.0.0.1',self.server.port,self.server.identity['fingerprint'],token=token)
    def tearDown(self): self.server.shutdown(); self.thread.join(); self.server.close(); self.tmp.cleanup()
    def submit(self,op):
        raw=encode_tensor('hidden','float32',[1],b'\x00\x00\x80\x3f'); digest=hashlib.sha256(raw).hexdigest(); self.admin.put_artifact(digest,raw)
        return self.admin.request('POST','/v1/jobs',{'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[{'task_id':'one','step_index':0,'fragment_id':'one','node_id':self.node['node_id'],'operation':op,'inputs':{'hidden':digest},'parameters':{},'outputs':{'hidden':{'dtype':'float32','shape':[1]}}}]},headers={'Idempotency-Key':op})
    def test_registered_callback_receives_actual_frame(self):
        job=self.submit('local')
        observed=[]
        def execute(task,inputs):
            observed.append(inputs['hidden'].payload)
            return {'hidden':encode_tensor('hidden','float32',[1],b'\x00\x00\x00\x40')}
        worker=Worker(self.client(self.node['token']),{'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}},{'local':execute})
        self.assertTrue(worker.run_once()); self.assertEqual(observed,[b'\x00\x00\x80\x3f'])
        self.assertEqual(self.admin.request('GET','/v1/jobs/'+job['job_id'])['state'],'succeeded')
    def test_unknown_operation_reports_failure_without_execution(self):
        job=self.submit('unknown'); worker=Worker(self.client(self.node['token']),{'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}},{})
        self.assertFalse(worker.run_once())
        self.assertEqual(self.admin.request('GET','/v1/jobs/'+job['job_id'])['state'],'failed')

    def test_heartbeat_continues_during_callback(self):
        self.submit('slow')
        heartbeat_seen=threading.Event()
        client=self.client(self.node['token']); original=client.request
        def request(method,path,*args,**kwargs):
            result=original(method,path,*args,**kwargs)
            if path=='/v1/heartbeat': heartbeat_seen.set()
            return result
        client.request=request
        def execute(task,inputs):
            self.assertTrue(heartbeat_seen.wait(2))
            return {'hidden':encode_tensor('hidden','float32',[1],inputs['hidden'].payload)}
        cap={'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}}
        self.assertTrue(Worker(client,cap,{'slow':execute},heartbeat_interval=.01).run_once())

    def test_executor_value_error_reports_terminal_failure(self):
        job=self.submit('invalid')
        def execute(task,inputs): raise ValueError('Invalid model inputs')
        worker=Worker(self.client(self.node['token']),{}, {'invalid':execute})
        self.assertFalse(worker.run_once())
        self.assertEqual(self.admin.request('GET','/v1/jobs/'+job['job_id'])['state'],'failed')
