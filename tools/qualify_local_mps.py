"""Opt-in local CPU/MPS process qualification; no H3 weights or physical cluster claim."""
import contextlib,io,json,os,subprocess,sys,tempfile,threading
from pathlib import Path
from heteromesh.cli import main,prepared_validator
from heteromesh.service import Coordinator
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'coordinator'; server=Coordinator(root,validate_job=prepared_validator(root)); t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
 common=['--state-dir',str(root),'--host','127.0.0.1','--port',str(server.port)]
 def call(args):
  out=io.StringIO()
  with contextlib.redirect_stdout(out): code=main(args)
  if code:raise RuntimeError('CLI qualification step failed')
  return json.loads(out.getvalue())
 try:
  ids=[];configs=[]
  for backend in ['numpy','mps']:
   invitation=Path(tmp)/(backend+'-pair.json');config=Path(tmp)/(backend+'-node.json')
   call(['pair',*common,'--out',str(invitation)])
   ids.append(call(['join','--pairing-file',str(invitation),'--config',str(config),'--backend',backend])['node_id']);configs.append(config)
  job=call(['submit-tiny',*common,'--nodes',*ids,'--steps','2'])
  env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()))
  for config in configs*2:
   result=subprocess.run([sys.executable,'-m','heteromesh','worker','--config',str(config),'--once'],env=env,capture_output=True,text=True,timeout=60)
   if result.returncode:raise RuntimeError('worker failed: '+result.stderr)
   report=json.loads(result.stdout)
   assert report['executed'],report
   print(json.dumps(report))
  report=call(['verify-tiny',*common,'--job',job['job_id']]);report.pop('job_id');report['evidence']='single-host-loopback-numpy-mps'
  print(json.dumps(report))
 finally:server.shutdown();t.join();server.close()
