"""HTTPS client that pins the DER certificate before transmitting credentials."""
from __future__ import annotations
import hashlib
import hmac
import http.client
import json
import re
import ssl

class PinError(ValueError): pass

class RemoteError(ValueError):
    def __init__(self,status,body):
        self.status=status; self.body=body
        super().__init__(f'Remote request failed: HTTP {status}')

class PinnedClient:
    def __init__(self,host,port,fingerprint,*,token=None,timeout=30):
        if not isinstance(fingerprint,str) or not re.fullmatch('[0-9a-f]{64}',fingerprint): raise PinError('Invalid certificate fingerprint')
        self.host=host; self.port=port; self.fingerprint=fingerprint; self.token=token; self.timeout=timeout
    def _connect(self):
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Trust is exclusively the out-of-band exact certificate fingerprint.
        context.check_hostname=False; context.verify_mode=ssl.CERT_NONE
        connection=http.client.HTTPSConnection(self.host,self.port,timeout=self.timeout,context=context)
        connection.connect()
        actual=hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest()
        if not hmac.compare_digest(actual,self.fingerprint):
            connection.close(); raise PinError('Coordinator certificate changed or is not trusted')
        return connection
    def request(self,method,path,body=None,*,headers=None):
        data=None if body is None else json.dumps(body,separators=(',',':'),allow_nan=False).encode()
        if data is not None and len(data)>65536: raise ValueError('Control request too large')
        request_headers=dict(headers or {})
        if self.token is not None: request_headers['Authorization']='Bearer '+self.token
        if data is not None: request_headers['Content-Type']='application/json'
        connection=self._connect()
        try:
            connection.request(method,path,body=data,headers=request_headers)
            response=connection.getresponse()
            raw=response.read(65537)
            if len(raw)>65536: raise ValueError('Control response too large')
            decoded=json.loads(raw)
            if response.status>=400: raise RemoteError(response.status,decoded)
            if decoded.get('protocol_version')!=1: raise ValueError('Protocol mismatch')
            return decoded
        finally: connection.close()

    @staticmethod
    def _scope(task):
        if task is None: return {}
        return {'X-Job-Id':task['job_id'],'X-Recovery-Epoch':str(task['recovery_epoch']),'X-Attempt-Id':task['attempt_id']}

    def put_artifact(self,digest,data,*,task=None,output_name=None):
        if len(data)>256*1024*1024+65540: raise ValueError('Artifact too large')
        headers=self._scope(task)
        if self.token is not None: headers['Authorization']='Bearer '+self.token
        if output_name is not None: headers['X-Output-Name']=output_name
        headers.update({'Content-Length':str(len(data)),'Content-Type':'application/octet-stream'})
        connection=self._connect()
        try:
            connection.putrequest('PUT','/v1/artifacts/'+digest)
            for key,value in headers.items(): connection.putheader(key,value)
            connection.endheaders()
            for start in range(0,len(data),4*1024*1024): connection.send(memoryview(data)[start:start+4*1024*1024])
            response=connection.getresponse(); raw=response.read(65537)
            if len(raw)>65536: raise ValueError('Control response too large')
            decoded=json.loads(raw)
            if response.status>=400: raise RemoteError(response.status,decoded)
            return decoded
        finally: connection.close()

    def get_artifact(self,digest,task=None):
        headers=self._scope(task)
        if self.token is not None: headers['Authorization']='Bearer '+self.token
        connection=self._connect()
        try:
            connection.request('GET','/v1/artifacts/'+digest,headers=headers)
            response=connection.getresponse()
            if response.status>=400: raise RemoteError(response.status,json.loads(response.read(65536)))
            raw=response.read(256*1024*1024+65541)
            if len(raw)>256*1024*1024+65540 or hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Invalid artifact')
            return raw
        finally: connection.close()
