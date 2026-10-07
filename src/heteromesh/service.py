"""Pinned-HTTPS coordinator with authenticated job and bounded artifact routes."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import re
import ssl
import tempfile
import struct
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .state import StateStore, StateError
from .security import SecurityStore, SecurityError, create_identity
from .protocol import copy_tensor, read_tensor, ProtocolError
from .manifests import validate_capabilities as validate_node_capabilities

MAX_CONTROL=65536
MAX_ARTIFACT=256*1024*1024+65540

def sync_directory(path):
    """Persist published directory entries on POSIX; Windows has a documented weaker guarantee."""
    if os.name == 'nt': return False
    fd=os.open(path,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
    try: os.fsync(fd)
    finally: os.close(fd)
    return True

class RequestError(ValueError):
    def __init__(self,status,code): self.status=status; self.code=code

class LimitedReader:
    def __init__(self,stream,length): self.stream=stream; self.remaining=length
    def read(self,n):
        if not self.remaining: return b''
        chunk=self.stream.read(min(n,self.remaining)); self.remaining-=len(chunk); return chunk

class Coordinator:
    def __init__(self,root,*,host='127.0.0.1',port=0,validate_job=None,validate_capabilities=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if os.name != 'nt': self.root.chmod(0o700)
        self.storage_durability='posix-fsync' if os.name!='nt' else 'process-crash-only'
        self.identity=create_identity(self.root/'identity')
        self.state=StateStore(self.root/'ledger.db')
        self.state.db.execute('CREATE TABLE IF NOT EXISTS output_grants (attempt TEXT NOT NULL, node TEXT NOT NULL, job TEXT NOT NULL, epoch INTEGER NOT NULL, name TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(attempt,name,digest))')
        self.security=SecurityStore(self.root/'identity.db')
        self.artifacts=self.root/'artifacts'; self.artifacts.mkdir(exist_ok=True)
        self.quarantine=self.root/'quarantine'; self.quarantine.mkdir(exist_ok=True)
        self.validate_job=validate_job
        self.validate_capabilities=validate_capabilities or validate_node_capabilities
        self.transfer_lock=threading.Lock(); self.transfers={}
        app=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args): pass
            def do_GET(self): self.dispatch()
            def do_POST(self): self.dispatch()
            def do_PUT(self): self.dispatch()
            def do_DELETE(self): self.dispatch()
            def reply(self,status,body):
                data=json.dumps(dict(protocol_version=1,**body),separators=(',',':'),allow_nan=False).encode()
                self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data))); self.send_header('Connection','close'); self.end_headers(); self.wfile.write(data); self.close_connection=True
            def dispatch(self):
                self.connection.settimeout(30)
                try: app.handle(self)
                except RequestError as exc: self.reply(exc.status,{'error':{'code':exc.code,'message':exc.code,'retryable':False}})
                except SecurityError: self.reply(401,{'error':{'code':'UNAUTHORIZED','message':'UNAUTHORIZED','retryable':False}})
                except StateError as exc: self.reply(409,{'error':{'code':exc.code,'message':exc.code,'retryable':False}})
                except (ProtocolError,ValueError,TypeError,KeyError): self.reply(400,{'error':{'code':'INVALID_REQUEST','message':'INVALID_REQUEST','retryable':False}})
                except (ConnectionError,TimeoutError): self.close_connection=True
                except OSError: self.reply(503,{'error':{'code':'STORAGE_UNAVAILABLE','message':'Storage operation failed','retryable':False}})
        self.http=ThreadingHTTPServer((host,port),Handler)
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(self.identity['cert'],self.identity['key'])
        self.http.socket=context.wrap_socket(self.http.socket,server_side=True)
        self.port=self.http.server_port
    def serve_forever(self): self.http.serve_forever()
    def shutdown(self): self.http.shutdown()
    def close(self): self.http.server_close(); self.state.close(); self.security.close()
    def _length(self,h,maximum):
        if h.headers.get_all('Transfer-Encoding'): raise RequestError(400,'INVALID_LENGTH')
        values=h.headers.get_all('Content-Length') or []
        if not values and h.command in ('GET','DELETE'): return 0
        if len(values)!=1 or not re.fullmatch('[0-9]+',values[0]): raise RequestError(400,'INVALID_LENGTH')
        length=int(values[0])
        if length>maximum: raise RequestError(413,'PAYLOAD_TOO_LARGE')
        return length
    def _json(self,h,length):
        raw=h.rfile.read(length)
        if len(raw)!=length: raise RequestError(400,'TRUNCATED')
        def pairs(values):
            result={}
            for k,v in values:
                if k in result: raise RequestError(400,'DUPLICATE_KEY')
                result[k]=v
            return result
        def reject(value): raise RequestError(400,'NONFINITE')
        result=json.loads(raw or b'{}',object_pairs_hook=pairs,parse_constant=reject)
        if not isinstance(result,dict): raise RequestError(400,'INVALID_REQUEST')
        return result
    def _identity(self,h):
        headers=h.headers.get_all('Authorization') or []
        if len(headers)!=1 or not headers[0].startswith('Bearer '): raise SecurityError()
        token=headers[0][7:]
        if hmac.compare_digest(token,self.identity['admin_token']): return 'admin'
        return self.security.authenticate(token)
    def _attempt(self,h,node,*,require_live=True):
        attempt_id=h.headers.get('X-Attempt-Id'); job_id=h.headers.get('X-Job-Id'); epoch=h.headers.get('X-Recovery-Epoch')
        if not epoch or not re.fullmatch('[0-9]+',epoch): raise RequestError(403,'FORBIDDEN')
        with self.state.lock:
            row=self.state.db.execute('SELECT * FROM attempts WHERE id=?',(attempt_id,)).fetchone()
            if row is None or row['node']!=node or row['job']!=job_id or row['epoch']!=int(epoch): raise RequestError(403,'FORBIDDEN')
            job=self.state._job(job_id)
            if job['epoch']!=row['epoch'] or job['state'] in ('cancelled','failed'): raise RequestError(403,'FORBIDDEN')
            live = row['state']=='leased' and row['deadline']>self.state.clock()
            committed = row['state']=='committed'
            if not live and (require_live or not committed): raise RequestError(403,'FORBIDDEN')
            task=json.loads(row['task'])
            task['_readable_artifacts'] = list(task['inputs'].values()) if live else list(json.loads(row['result'])['outputs'].values())
            return task
    def _semaphore(self,node):
        with self.transfer_lock: return self.transfers.setdefault(node,threading.BoundedSemaphore(2))
    def handle(self,h):
        artifact=re.fullmatch('/v1/artifacts/([0-9a-f]{64})',h.path)
        length=self._length(h,MAX_ARTIFACT if artifact else MAX_CONTROL)
        if h.path=='/v1/health' and h.command=='GET': h.reply(200,{'status':'ready','storage_durability':self.storage_durability}); return
        if h.path=='/v1/nodes/register' and h.command=='POST':
            headers=h.headers.get_all('Authorization') or []
            if len(headers)!=1 or not headers[0].startswith('Bearer '): raise SecurityError()
            body=self._json(h,length); cap=body['capabilities']
            if self.validate_capabilities is not None: self.validate_capabilities(cap)
            h.reply(200,self.security.register(headers[0][7:],cap)); return
        identity=self._identity(h)
        if artifact:
            self._artifact(h,identity,artifact[1],length); return
        body=self._json(h,length) if length else {}
        if h.path=='/v1/pairing' and h.command=='POST':
            self._admin(identity); h.reply(200,self.security.create_pairing()); return
        if h.path=='/v1/nodes' and h.command=='GET':
            self._admin(identity); h.reply(200,{'nodes':self.security.nodes()}); return
        if h.path.startswith('/v1/nodes/') and h.command=='DELETE':
            self._admin(identity); node=h.path.split('/')[-1]; self.security.revoke(node); self.state.invalidate_node(node); h.reply(200,{'revoked':True}); return
        if h.path=='/v1/heartbeat' and h.command=='POST':
            self._node(identity)
            cap=body['capabilities']
            if self.validate_capabilities is not None: self.validate_capabilities(cap)
            self.security.heartbeat(identity,cap); h.reply(200,{'accepted':True}); return
        if h.path=='/v1/work/lease' and h.command=='POST':
            self._node(identity); h.reply(200,{'task':self.state.lease(identity)}); return
        if h.path=='/v1/work/result' and h.command=='POST':
            self._node(identity)
            with self.state.lock:
                row=self.state.db.execute('SELECT task FROM attempts WHERE id=?',(body.get('attempt_id'),)).fetchone()
                if row is None: raise StateError('INVALID_LEASE')
                task=json.loads(row['task'])
            for name,digest in body['outputs'].items():
                with self.state.lock:
                    grant=self.state.db.execute('SELECT 1 FROM output_grants WHERE attempt=? AND node=? AND job=? AND epoch=? AND name=? AND digest=?', (body.get('attempt_id'),identity,body.get('job_id'),body.get('recovery_epoch'),name,digest)).fetchone()
                if grant is None: raise RequestError(403,'UNAUTHORIZED_OUTPUT')
                self._verify_artifact(digest,task['outputs'][name],name)
            h.reply(200,self.state.commit_result(identity,body,body['outputs'])); return
        if h.path=='/v1/work/error' and h.command=='POST':
            self._node(identity); h.reply(200,self.state.report_error(identity,body,body['code'])); return
        if h.path=='/v1/jobs' and h.command=='POST':
            self._admin(identity)
            if self.validate_job is None: raise RequestError(422,'UNSUPPORTED_PROFILE')
            self.state.validate_request(body)
            self.validate_job(body,self.security.nodes())
            h.reply(200,self.state.submit_job(h.headers.get('Idempotency-Key'),body)); return
        match=re.fullmatch('/v1/jobs/([0-9a-f]{32})(?:/(cancel|pause|resume))?',h.path)
        if match:
            self._admin(identity)
            if match[2] and h.command=='POST': h.reply(200,getattr(self.state,match[2])(match[1])); return
            if not match[2] and h.command=='GET': h.reply(200,self.state.get_job(match[1])); return
        raise RequestError(404,'NOT_FOUND')
    def _admin(self,identity):
        if identity!='admin': raise RequestError(403,'FORBIDDEN')
    def _node(self,identity):
        if identity=='admin': raise RequestError(403,'FORBIDDEN')
    @staticmethod
    def _finite_artifact(path,header):
        dtype=header['dtype']
        if dtype not in ('float16','bfloat16','float32'): return
        mask=0x7f800000 if dtype=='float32' else (0x7c00 if dtype=='float16' else 0x7f80)
        word='<u4' if dtype=='float32' else '<u2'
        with Path(path).open('rb') as source:
            header_bytes=struct.unpack('>I',source.read(4))[0]
            source.seek(4+header_bytes)
            while chunk:=source.read(65536):
                bits=np.frombuffer(chunk,dtype=word)
                if np.any((bits & mask)==mask): raise RequestError(400,'NONFINITE_OUTPUT')

    def _verify_artifact(self,digest,spec,name):
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest): raise RequestError(400,'INVALID_ARTIFACT')
        path=self.artifacts/digest
        if not path.is_file(): raise RequestError(400,'MISSING_ARTIFACT')
        with path.open('rb') as source:
            class NullSink:
                def write(self,data): return len(data)
            meta=copy_tensor(source,NullSink())
        if meta['artifact_sha256']!=digest: raise RequestError(400,'INVALID_ARTIFACT')
        if meta['header']['name']!=name or any(meta['header'][key]!=spec[key] for key in ('dtype','shape')): raise RequestError(400,'OUTPUT_MISMATCH')
        self._finite_artifact(path,meta['header'])
    def _artifact(self,h,node,digest,length):
        task=None
        if node!='admin':
            task=self._attempt(h,node,require_live=h.command!='GET')
            if h.command=='GET' and digest not in task['_readable_artifacts']: raise RequestError(403,'FORBIDDEN')
            if h.command=='PUT' and h.headers.get('X-Output-Name') not in task['outputs']: raise RequestError(403,'FORBIDDEN')
        semaphore=self._semaphore(node)
        if not semaphore.acquire(blocking=False): raise RequestError(503,'TRANSFER_BUSY')
        try:
            if h.command=='GET':
                path=self.artifacts/digest
                if not path.is_file(): raise RequestError(404,'NOT_FOUND')
                h.send_response(200); h.send_header('Content-Type','application/octet-stream'); h.send_header('Content-Length',str(path.stat().st_size)); h.send_header('Connection','close'); h.end_headers()
                with path.open('rb') as source:
                    while chunk:=source.read(4*1024*1024): h.wfile.write(chunk)
                h.close_connection=True; return
            if h.command!='PUT': raise RequestError(404,'NOT_FOUND')
            fd,tmp=tempfile.mkstemp(dir=self.quarantine)
            try:
                with os.fdopen(fd,'wb') as sink:
                    meta=copy_tensor(LimitedReader(h.rfile,length),sink)
                    if meta['artifact_sha256']!=digest: raise RequestError(400,'INVALID_ARTIFACT')
                    if task is not None:
                        name=h.headers['X-Output-Name']; spec=task['outputs'][name]
                        if meta['header']['name']!=name or any(meta['header'][key]!=spec[key] for key in ('dtype','shape')): raise RequestError(400,'OUTPUT_MISMATCH')
                    sink.flush(); os.fsync(sink.fileno())
                if task is not None:
                    self._finite_artifact(tmp,meta['header'])
                    self._attempt(h,node)
                os.replace(tmp,self.artifacts/digest)
                sync_directory(self.artifacts)
                if task is not None:
                    with self.state._tx():
                        self._attempt(h,node)
                        self.state.db.execute('INSERT OR IGNORE INTO output_grants VALUES(?,?,?,?,?,?)', (task['attempt_id'],node,task['job_id'],task['recovery_epoch'],h.headers['X-Output-Name'],digest))
                h.reply(200,{'artifact_sha256':digest})
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        finally: semaphore.release()
