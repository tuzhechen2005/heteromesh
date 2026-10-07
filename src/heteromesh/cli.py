"""Local command-line onboarding and genuine dependent-block integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import ssl
import sys
import threading
import time

import numpy as np
import psutil

from .client import PinnedClient
from .demo import OPERATION, PROFILE, build_plan, sample, validate_plan
from .protocol import decode_tensor
from .runtime import TensorExecutor, arrays_to_frames
from .service import Coordinator
from .worker import Worker


def private_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:
        json.dump(data,handle,ensure_ascii=False,allow_nan=False,indent=2)
        handle.flush();os.fsync(handle.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepared_validator(root):
    root=Path(root)
    def validate(request,nodes):
        artifacts=validate_plan(request,nodes)
        for digest,expected in artifacts.items():
            path=root/'artifacts'/digest
            # This bounded built-in profile uses tiny files, not H3-sized shards.
            if not path.is_file() or path.stat().st_size!=len(expected) or path.read_bytes()!=expected:
                raise ValueError('missing or corrupt initial profile artifact')
    return validate


def capabilities(backend,host_budget_mib=1024,gpu_budget_mib=1024):
    if min(host_budget_mib,gpu_budget_mib)<=0:raise ValueError('budgets must be positive')
    executor=TensorExecutor(backend)  # Explicitly verifies the selected accelerator exists.
    host=min(host_budget_mib*1024*1024,psutil.virtual_memory().available//2)
    unified=platform.system()=='Darwin'
    accelerator=None
    if backend=='cuda':
        free,_=executor.torch.cuda.mem_get_info()
        accelerator=min(gpu_budget_mib*1024*1024,int(free)//2)
    cap={'protocol_version':1,'platform':{'Darwin':'macos','Windows':'windows','Linux':'linux'}[platform.system()],
         'runtime':backend,'backend':'cpu' if backend in ('numpy','torch-cpu') else backend,
         'supported_ops':[OPERATION],'wire_dtypes':['float32'],'compute_dtypes':['float32'],
         'memory':{'unified':unified,'host_budget_bytes':host,'accelerator_budget_bytes':accelerator}}
    return cap,executor


def admin_client(args):
    identity=Path(args.state_dir)/'identity'
    der=ssl.PEM_cert_to_DER_cert((identity/'cert.pem').read_text(encoding='utf-8'))
    return PinnedClient(args.host,args.port,hashlib.sha256(der).hexdigest(),
                        token=(identity/'admin.token').read_text(encoding='utf-8').strip())


def parser():
    p=argparse.ArgumentParser(prog='heteromesh',description='Experimental local heterogeneous inference')
    subs=p.add_subparsers(dest='command',required=True)
    for command in ['coordinator','pair','nodes','submit-tiny','status','cancel','pause','resume','verify-tiny']:
        sub=subs.add_parser(command)
        sub.add_argument('--state-dir',default=str(Path.home()/'.heteromesh'/'coordinator'))
        sub.add_argument('--host',default='127.0.0.1')
        sub.add_argument('--port',type=int,default=7443)
        if command=='pair':sub.add_argument('--out',required=True)
        if command=='submit-tiny':
            sub.add_argument('--nodes',nargs='+',required=True)
            sub.add_argument('--steps',type=int,default=2)
        if command in {'status','cancel','pause','resume','verify-tiny'}:sub.add_argument('--job',required=True)
    join=subs.add_parser('join')
    join.add_argument('--pairing-file',required=True)
    join.add_argument('--config',required=True)
    join.add_argument('--backend',choices=['numpy','torch-cpu','mps','cuda'],required=True)
    join.add_argument('--host-budget-mib',type=int,default=1024)
    join.add_argument('--gpu-budget-mib',type=int,default=1024)
    worker=subs.add_parser('worker')
    worker.add_argument('--config',required=True)
    worker.add_argument('--once',action='store_true')
    return p


def _execute(args):
    command=args.command
    if command=='coordinator':
        server=Coordinator(args.state_dir,host=args.host,port=args.port,validate_job=prepared_validator(args.state_dir))
        print(json.dumps({'status':'ready','port':server.port,'fingerprint':server.identity['fingerprint']}),flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.close()
        return 0
    if command=='join':
        if Path(args.config).exists():raise FileExistsError('node config already exists')
        invitation=read_json(args.pairing_file)
        if invitation['expires']<=time.time():raise ValueError('pairing invitation expired')
        cap,_=capabilities(args.backend,args.host_budget_mib,args.gpu_budget_mib)
        client=PinnedClient(invitation['host'],invitation['port'],invitation['fingerprint'],token=invitation['token'])
        node=client.request('POST','/v1/nodes/register',{'capabilities':cap})
        private_json(args.config,{'host':invitation['host'],'port':invitation['port'],'fingerprint':invitation['fingerprint'],
                                 'token':node['token'],'node_id':node['node_id'],'backend':args.backend,
                                 'host_budget_mib':args.host_budget_mib,'gpu_budget_mib':args.gpu_budget_mib})
        print(json.dumps({'node_id':node['node_id'],'backend':args.backend,'config_saved':True}))
        return 0
    if command=='worker':
        config=read_json(args.config)
        client=PinnedClient(config['host'],config['port'],config['fingerprint'],token=config['token'])
        cap,executor=capabilities(config['backend'],config['host_budget_mib'],config['gpu_budget_mib'])
        worker=Worker(client,cap,{OPERATION:executor})
        try:
            while True:
                client.request('POST','/v1/heartbeat',{'capabilities':cap})
                ran=worker.run_once()
                if args.once:
                    print(json.dumps({'executed':ran,'backend':config['backend']}));return 0
                time.sleep(1)
        except KeyboardInterrupt:return 0
    client=admin_client(args)
    if command=='pair':
        if Path(args.out).exists():raise FileExistsError('invitation file already exists')
        pair=client.request('POST','/v1/pairing',{})
        private_json(args.out,dict(pair,host=args.host,port=args.port,fingerprint=client.fingerprint))
        print(json.dumps({'invitation_saved':True,'expires':pair['expires']}))
    elif command=='nodes':
        print(json.dumps(client.request('GET','/v1/nodes'),indent=2))
    elif command=='submit-tiny':
        available=client.request('GET','/v1/nodes')['nodes'];by_id={n['node_id']:n for n in available}
        request,artifacts,_=build_plan([by_id[n] for n in args.nodes],steps=args.steps)
        validate_plan(request,available)
        for digest,raw in artifacts.items():client.put_artifact(digest,raw)
        job=client.request('POST','/v1/jobs',request,headers={'Idempotency-Key':secrets.token_hex(16)})
        private_json(Path(args.state_dir)/'submitted'/(job['job_id']+'.json'),request)
        print(json.dumps(job))
    elif command in {'status','cancel','pause','resume'}:
        method='GET' if command=='status' else 'POST'
        suffix='' if command=='status' else '/'+command
        print(json.dumps(client.request(method,'/v1/jobs/'+args.job+suffix,None if method=='GET' else {})))
    elif command=='verify-tiny':
        # Job IDs come from the coordinator; reject path traversal in local records.
        import re
        if not re.fullmatch('[0-9a-f]{32}',args.job):raise ValueError('invalid job ID')
        job=client.request('GET','/v1/jobs/'+args.job)
        if job['state']!='succeeded':raise ValueError('job has not succeeded')
        request=read_json(Path(args.state_dir)/'submitted'/(args.job+'.json'))
        frame=decode_tensor(client.get_artifact(job['outputs']['hidden']))
        if frame.header['dtype']!='float32' or frame.header['shape']!=[3,4]:raise ValueError('final shape mismatch')
        actual=np.frombuffer(frame.payload,dtype='<f4').reshape(3,4)
        x=sample();reference=TensorExecutor('numpy')
        for task in request['tasks']:
            output=decode_tensor(reference(task,arrays_to_frames(x))['hidden'])
            x['hidden']=np.frombuffer(output.payload,dtype='<f4').reshape(3,4).copy()
        match=bool(np.allclose(actual,x['hidden'],atol=float(PROFILE['atol']),rtol=float(PROFILE['rtol'])))
        print(json.dumps({'job_id':args.job,'numerical_match':match,'max_absolute_error':float(np.max(np.abs(actual-x['hidden']))),
                          'evidence':'local-process-or-user-connected-nodes','profile':PROFILE['id'],'h3_validated':False}))
        return 0 if match else 1
    return 0


def main(argv=None):
    args=parser().parse_args(argv)
    try:return _execute(args)
    except (ValueError,KeyError,OSError,RuntimeError) as exc:
        # Do not echo credential-bearing objects or server-controlled error bodies.
        print(f'{type(exc).__name__}: command failed; check configuration and node availability.',file=sys.stderr)
        return 1
