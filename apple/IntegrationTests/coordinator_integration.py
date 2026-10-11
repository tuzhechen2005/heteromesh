"""Real Python HTTPS coordinator + real Swift worker, synthetic tiny weights only."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
from heteromesh.client import PinnedClient
from heteromesh.service import Coordinator
from heteromesh.protocol import encode_tensor, decode_tensor


def main():
    binary = Path(__file__).resolve().parents[1] / '.build/debug/apple-node'
    with tempfile.TemporaryDirectory() as tmp:
        # Exact synthetic operation below is the only profile used by this integration test.
        def admit(request, nodes):
            assert len(request['tasks']) == 1
            assert request['tasks'][0]['operation'] == 'tiny_transformer_block_v1'
        server = Coordinator(Path(tmp),validate_job=admit)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        admin = PinnedClient('127.0.0.1',server.port,server.identity['fingerprint'],token=server.identity['admin_token'])
        pair = admin.request('POST','/v1/pairing',{})
        env = dict(os.environ,HETEROMESH_PAIRING_TOKEN=pair['token'])
        worker = subprocess.Popen([str(binary),'run','https://127.0.0.1:'+str(server.port),server.identity['fingerprint']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+20
            while time.monotonic()<deadline:
                nodes = admin.request('GET','/v1/nodes')['nodes']
                if nodes: break
                if worker.poll() is not None: raise AssertionError(worker.communicate()[1].decode())
                time.sleep(.05)
            assert nodes, 'Swift registration timed out'
            inputs = {}
            specs = {'hidden':([1,2],[1.,3.]),'condition':([1,2],[0.,0.])}
            specs.update({n:([2,2],[1.,0.,0.,1.]) for n in ['wq','wk','wv','wo','w1','w2']})
            specs.update({n:([2],[0.,0.]) for n in ['b1','b2','ln1_bias','ln2_bias']})
            specs.update({n:([2],[1.,1.]) for n in ['ln1_weight','ln2_weight']})
            for name,(shape,values) in specs.items():
                raw = encode_tensor(name,'float32',shape,struct.pack('<'+'f'*len(values),*values))
                digest = hashlib.sha256(raw).hexdigest(); admin.put_artifact(digest,raw);inputs[name]=digest
            job = admin.request('POST','/v1/jobs',{'manifest_digest':'a'*64,'profile_digest':'b'*64,'tasks':[{'task_id':'one','step_index':0,'fragment_id':'one','node_id':nodes[0]['node_id'],'operation':'tiny_transformer_block_v1','inputs':inputs,'parameters':{'epsilon':'0.00001'},'outputs':{'hidden':{'dtype':'float32','shape':[1,2]}}}]},headers={'Idempotency-Key':'swift-real-compute'})
            while time.monotonic()<deadline:
                state = admin.request('GET','/v1/jobs/'+job['job_id'])
                if state['state'] in ('succeeded','failed'): break
                if worker.poll() is not None: raise AssertionError(worker.communicate()[1].decode())
                time.sleep(.1)
            assert state['state']=='succeeded',(state,worker.communicate(timeout=5)[1].decode() if state['state']=='failed' else '')
            with server.state.lock:
                import json
                result = json.loads(server.state.db.execute('SELECT result FROM attempts WHERE state=?',('committed',)).fetchone()['result'])
            raw = admin.get_artifact(result['outputs']['hidden'])
            vals = struct.unpack('<ff',decode_tensor(raw).payload)
            assert abs(vals[0])<1e-4 and abs(vals[1]-5)<1e-4,vals
            print('PASS: Python coordinator -> Swift real Transformer -> validated output via pinned HTTPS')
        finally:
            worker.terminate(); worker.communicate(timeout=10)
            server.shutdown();thread.join();server.close()

if __name__=='__main__': main()
