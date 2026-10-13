"""HTTPS client that pins the DER certificate before transmitting credentials."""
from __future__ import annotations
import hashlib
import hmac
import http.client
import json
import re
import ssl
import struct
from .protocol import MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, CHUNK_BYTES, _read_exact, parse_json, validate_header

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

    def get_artifact(self,digest,task=None,*,max_frame_bytes=None,max_payload_bytes=MAX_PAYLOAD_BYTES,expected_name=None,allowed_dtypes=None):
        if max_frame_bytes is not None and (type(max_frame_bytes) is not int or max_frame_bytes<0 or type(max_payload_bytes) is not int or max_payload_bytes<0):
            raise ValueError('Invalid local artifact budget')
        headers=self._scope(task)
        if self.token is not None: headers['Authorization']='Bearer '+self.token
        connection=self._connect()
        try:
            connection.request('GET','/v1/artifacts/'+digest,headers=headers)
            response=connection.getresponse()
            if response.status>=400:
                error=response.read(65537)
                if len(error)>65536:raise ValueError('Control response too large')
                raise RemoteError(response.status,json.loads(error))
            if max_frame_bytes is None:
                raw=response.read(256*1024*1024+65541)
            else:
                frame_limit=min(max_frame_bytes,MAX_PAYLOAD_BYTES+MAX_HEADER_BYTES+4)
                payload_limit=min(max_payload_bytes,MAX_PAYLOAD_BYTES)
                fields=response.getheaders()
                lengths=[value for key,value in fields if key.lower()=='content-length']
                if any(key.lower()=='transfer-encoding' for key,_ in fields) or len(lengths)!=1 or re.fullmatch('[0-9]+',lengths[0]) is None:
                    raise ValueError('Invalid artifact HTTP framing')
                # Avoid giant integer parsing as well as oversized body reads.
                digits=lengths[0].lstrip('0') or '0'
                if len(digits)>len(str(frame_limit)) or int(digits)>frame_limit:
                    raise ValueError('Artifact exceeds installed frame budget')
                length=int(digits)
                if length<4:raise ValueError('Truncated artifact prefix')
                prefix=_read_exact(response,4)
                header_size=struct.unpack('>I',prefix)[0]
                if not 0<header_size<=MAX_HEADER_BYTES or header_size+4>length:
                    raise ValueError('Invalid artifact header length')
                header_raw=_read_exact(response,header_size)
                header=parse_json(header_raw)
                validate_header(header,max_payload_bytes=payload_limit)
                if expected_name is not None and header['name']!=expected_name:raise ValueError('Artifact input name differs')
                if allowed_dtypes is not None and header['dtype'] not in allowed_dtypes:raise ValueError('Artifact input dtype differs')
                if 4+header_size+header['payload_bytes']!=length:
                    raise ValueError('Artifact HTTP/tensor length differs')
                # Hash exact wire bytes, including legal noncanonical header JSON.
                data=bytearray(prefix);data.extend(header_raw)
                payload_hash=hashlib.sha256();remaining=header['payload_bytes']
                while remaining:
                    chunk=response.read(min(CHUNK_BYTES,remaining))
                    if not chunk or len(chunk)>min(CHUNK_BYTES,remaining):raise ValueError('Truncated/invalid artifact payload')
                    data.extend(chunk);payload_hash.update(chunk);remaining-=len(chunk)
                if response.read(1):raise ValueError('Trailing artifact bytes')
                if payload_hash.hexdigest()!=header['sha256']:raise ValueError('Artifact payload hash differs')
                raw=bytes(data)
            if len(raw)>256*1024*1024+65540 or hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Invalid artifact')
            return raw
        finally: connection.close()
