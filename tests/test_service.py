import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from heteromesh.service import Coordinator
from heteromesh.client import PinnedClient, RemoteError
from heteromesh.protocol import encode_tensor
from unittest.mock import patch
import subprocess
import sys


class CoordinatorStartupTests(unittest.TestCase):
    def test_binding_does_not_require_reverse_dns(self):
        for host in ('127.0.0.1','localhost'):
            with self.subTest(host=host),tempfile.TemporaryDirectory() as root:
                with patch('socket.getfqdn',side_effect=RuntimeError('reverse DNS unavailable')):
                    app=Coordinator(root,host=host)
                thread=threading.Thread(target=app.serve_forever,daemon=True);thread.start()
                try:
                    assert app.http.server_name==app.http.server_address[0]
                    assert app.port==app.http.server_address[1]>0
                    client=PinnedClient('127.0.0.1',app.port,app.identity['fingerprint'],token=app.identity['admin_token'])
                    assert client.request('GET','/v1/nodes')['nodes']==[]
                finally:app.shutdown();thread.join();app.close()

    def test_restore_missing_optional_runtime_is_structured_and_read_only(self):
        script='''import importlib.abc,sys,threading,json
from heteromesh.service import Coordinator
from heteromesh.client import PinnedClient,RemoteError
blocked=[]
class MissingOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='heteromesh.h3_recovery':
   # E1 import-boundary injection exercises either exact missing dependency
   # name even in .[test] environments where neither optional package exists.
   blocked.append(sys.argv[2])
   raise ModuleNotFoundError('optional runtime absent',name=sys.argv[2])
sys.meta_path.insert(0,MissingOptional())
a=Coordinator(sys.argv[1]);a.http.handle_error=lambda *_:None
thread=threading.Thread(target=a.serve_forever,daemon=True);thread.start()
client=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=a.identity['admin_token'])
request={'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[{'task_id':'one','fragment_id':'one','step_index':0,'node_id':'pc','operation':'tiny','parameters':{},'inputs':{},'outputs':{'hidden':{'dtype':'float32','shape':[1]}}}]}
ordinary=a.state.submit_job('ordinary',request)['job_id']
h3=a.state.submit_job('h3',dict(request,h3_recovery=True))['job_id']
before=list(a.state.db.iterdump())
try:
 for job,expected in [(ordinary,'CHECKPOINT_REQUIRED'),('f'*32,'NOT_FOUND'),(h3,'UNSUPPORTED_RUNTIME')]:
  try:client.request('POST','/v1/jobs/'+job+'/restore',{'request_id':'r','expected_epoch':0})
  except RemoteError as exc:
   assert exc.body['error']['code']==expected,(exc.body,expected)
  else:raise AssertionError('restore unexpectedly succeeded')
  if job!=h3:assert blocked==[],'non-H3 restore imported optional runtime'
 assert blocked==[sys.argv[2]] and a._h3 is None
 assert list(a.state.db.iterdump())==before
 print('structured optional-runtime denial; authority unchanged')
finally:a.shutdown();thread.join();a.close()
'''
        for missing in ('torch','diffusers'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as root:
                result=subprocess.run([sys.executable,'-c',script,root,missing],capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_unrelated_import_failure_is_not_optional_runtime_denial(self):
        import builtins
        original=builtins.__import__
        def imported(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='h3_recovery' and level==1:
                raise ModuleNotFoundError('internal module defect',name='internal_dependency')
            return original(name,globals,locals,fromlist,level)
        with tempfile.TemporaryDirectory() as root:
            app=Coordinator(root);before=list(app.state.db.iterdump())
            try:
                with patch('builtins.__import__',side_effect=imported),self.assertRaises(ModuleNotFoundError) as caught:
                    _=app.h3
                self.assertEqual(caught.exception.name,'internal_dependency')
                self.assertIsNone(app._h3)
                self.assertEqual(list(app.state.db.iterdump()),before)
            finally:app.close()

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.server=Coordinator(Path(self.tmp.name),validate_job=lambda request,nodes: None)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.admin=self.client(self.server.identity['admin_token'])
    def tearDown(self):
        self.server.shutdown(); self.thread.join(); self.server.close(); self.tmp.cleanup()
    def client(self,token=None):
        return PinnedClient('127.0.0.1',self.server.port,self.server.identity['fingerprint'],token=token)
    def pair(self):
        token=self.admin.request('POST','/v1/pairing',{})['token']
        node=self.client(token).request('POST','/v1/nodes/register',{'capabilities': {'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}}})
        return node,self.client(node['token'])
    def test_pair_auth_revoke(self):
        with self.assertRaises(RemoteError): self.client().request('GET','/v1/nodes')
        node,client=self.pair()
        self.assertIsNone(client.request('POST','/v1/work/lease',{})['task'])
        self.admin.request('DELETE','/v1/nodes/'+node['node_id'])
        with self.assertRaises(RemoteError): client.request('POST','/v1/work/lease',{})
    def test_full_job_artifact_and_scoped_access(self):
        node,client=self.pair(); other,outsider=self.pair()
        data=encode_tensor('hidden','float32',[1],b'\x00\x00\x80\x3f')
        digest=hashlib.sha256(data).hexdigest()
        self.admin.put_artifact(digest,data)
        request={'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[{'task_id':'one','step_index':0,'fragment_id':'one','node_id':node['node_id'],'operation':'tiny','parameters':{},'inputs':{'hidden':digest},'outputs':{'hidden':{'dtype':'float32','shape':[1]}}}]}
        job=self.admin.request('POST','/v1/jobs',request,headers={'Idempotency-Key':'job'})
        task=client.request('POST','/v1/work/lease',{})['task']
        self.assertEqual(client.get_artifact(digest,task),data)
        with self.assertRaises(RemoteError): outsider.get_artifact(digest,task)
        nan_output=encode_tensor('hidden','float32',[1],b'\x00\x00\xc0\x7f')
        with self.assertRaises(RemoteError): client.put_artifact(hashlib.sha256(nan_output).hexdigest(),nan_output,task=task,output_name='hidden')
        output=encode_tensor('hidden','float32',[1],b'\x00\x00\x00\x40')
        out_digest=hashlib.sha256(output).hexdigest()
        client.put_artifact(out_digest,output,task=task,output_name='hidden')
        client.request('POST','/v1/work/result',dict(task,outputs={'hidden':out_digest}))
        self.assertEqual(self.admin.request('GET','/v1/jobs/'+job['job_id'])['state'],'succeeded')
        self.assertEqual(client.get_artifact(out_digest,task),output)
    def test_declared_hash_and_output_shape_are_verified(self):
        data=encode_tensor('hidden','float32',[1],b'\x00'*4)
        with self.assertRaises(RemoteError): self.admin.put_artifact('f'*64,data)
        self.assertEqual(list(self.server.artifacts.iterdir()),[])

    def test_malformed_capabilities_rejected_before_pair_consumed(self):
        token=self.admin.request('POST','/v1/pairing',{})['token']
        with self.assertRaises(RemoteError): self.client(token).request('POST','/v1/nodes/register',{'capabilities':{'backend':'cpu'}})

    def test_result_cannot_launder_unrelated_global_artifact(self):
        node,client=self.pair()
        secret=encode_tensor('hidden','float32',[1],b'\x00\x00\x80\x3f')
        digest=hashlib.sha256(secret).hexdigest()
        self.admin.put_artifact(digest,secret)
        request={'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[{'task_id':'one','step_index':0,'fragment_id':'one','node_id':node['node_id'],'operation':'tiny','parameters':{},'inputs':{},'outputs':{'hidden':{'dtype':'float32','shape':[1]}}}]}
        self.admin.request('POST','/v1/jobs',request,headers={'Idempotency-Key':'scope'})
        task=client.request('POST','/v1/work/lease',{})['task']
        with self.assertRaises(RemoteError): client.get_artifact(digest,task)
        with self.assertRaises(RemoteError): client.request('POST','/v1/work/result',dict(task,outputs={'hidden':digest}))
        with self.assertRaises(RemoteError): client.get_artifact(digest,task)

    def test_streaming_finite_check_respects_each_float_format(self):
        from heteromesh.service import RequestError
        from heteromesh.protocol import decode_tensor
        cases=[('float16',b'\xff\x7b',b'\x00\x7c'),('bfloat16',b'\x7f\x7f',b'\x80\x7f'),('float32',b'\xff\xff\x7f\x7f',b'\x00\x00\x80\x7f')]
        for dtype,finite,nonfinite in cases:
            with self.subTest(dtype=dtype):
                path=Path(self.tmp.name)/'finite-check'
                raw=encode_tensor('hidden',dtype,[1],finite); path.write_bytes(raw)
                Coordinator._finite_artifact(path,decode_tensor(raw).header)
                raw=encode_tensor('hidden',dtype,[1],nonfinite); path.write_bytes(raw)
                with self.assertRaises(RequestError): Coordinator._finite_artifact(path,decode_tensor(raw).header)

    def test_directory_sync_failure_prevents_upload_ack(self):
        from unittest.mock import patch
        raw=encode_tensor('hidden','float32',[1],b'\x00'*4)
        digest=hashlib.sha256(raw).hexdigest()
        with patch('heteromesh.service.sync_directory',side_effect=OSError('storage sync failed'),create=True):
            with self.assertRaises(RemoteError): self.admin.put_artifact(digest,raw)
        with self.server.state.lock:
            self.assertEqual(self.server.state.db.execute('SELECT count(*) FROM output_grants').fetchone()[0],0)
    def test_runtime_directory_is_private_on_posix(self):
        import os
        if os.name=='nt': return
        root=Path(self.tmp.name)/'permissive'; root.mkdir(mode=0o755); root.chmod(0o755)
        app=Coordinator(root)
        try: self.assertEqual(root.stat().st_mode & 0o777,0o700)
        finally: app.close()
