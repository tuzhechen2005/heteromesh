"""Pull worker executes only a caller-installed operation registry."""
import hashlib
import threading
from dataclasses import dataclass
from .client import RemoteError
from .protocol import decode_tensor, validate_finite, ProtocolError, MAX_PAYLOAD_BYTES, MAX_HEADER_BYTES

H3_OPERATIONS=frozenset('h3_'+role+'_v1' for role in ('prefix','range','suffix','scheduler_pair'))

@dataclass(frozen=True)
class InputPolicy:
    """Trusted local installation limits, never supplied by a remote task."""
    max_input_bytes: int
    max_payload_bytes: int
    input_dtypes: tuple

    def __post_init__(self):
        if any(type(n) is not int or n<0 for n in (self.max_input_bytes,self.max_payload_bytes)):
            raise ProtocolError('Invalid installed input budget')
        object.__setattr__(self,'input_dtypes',tuple((name,tuple(dtypes)) for name,dtypes in self.input_dtypes))
        if len({name for name,_ in self.input_dtypes})!=len(self.input_dtypes):raise ProtocolError('Duplicate installed input name')

class Worker:
    def __init__(self,client,capabilities,registry,*,heartbeat_interval=5):
        self.client=client; self.capabilities=capabilities; self.registry=dict(registry)
        self.heartbeat_interval=heartbeat_interval; self.heartbeat_error=None
    def _heartbeat(self,stop):
        while not stop.wait(self.heartbeat_interval):
            try: self.client.request('POST','/v1/heartbeat',{'capabilities':self.capabilities})
            except Exception as exc:
                self.heartbeat_error=type(exc).__name__
                return
    def run_once(self):
        task=self.client.request('POST','/v1/work/lease',{})['task']
        if task is None: return False
        operation=self.registry.get(task['operation'])
        if operation is None:
            self.client.request('POST','/v1/work/error',dict(task,code='UNSUPPORTED_OPERATOR'))
            return False
        stop=threading.Event(); heartbeat=threading.Thread(target=self._heartbeat,args=(stop,),daemon=True)
        heartbeat.start()
        try:
            policy_method=getattr(operation,'input_policy',None)
            if policy_method is None and task['operation'] in H3_OPERATIONS:
                raise ProtocolError('H3 operation has no local input policy')
            policy=None if policy_method is None else policy_method(task)
            if policy is not None and type(policy) is not InputPolicy:raise ProtocolError('Invalid installed input policy')
            if policy_method is not None and policy is None:raise ProtocolError('Missing installed input policy')
            inputs={};held_bytes=0;held_payload=0
            dtypes={} if policy is None else dict(policy.input_dtypes)
            if policy is not None and set(task['inputs'])!=set(dtypes):raise ProtocolError('Input policy names differ')
            for name,digest in task['inputs'].items():
                if policy is None:
                    raw=self.client.get_artifact(digest,task)
                    frame=decode_tensor(raw)
                else:
                    remaining=min(MAX_PAYLOAD_BYTES,policy.max_payload_bytes-held_payload)
                    frame_limit=min(MAX_PAYLOAD_BYTES+MAX_HEADER_BYTES+4,
                        policy.max_input_bytes-held_bytes,remaining+MAX_HEADER_BYTES+4)
                    raw=self.client.get_artifact(digest,task,max_frame_bytes=frame_limit,
                        max_payload_bytes=remaining,expected_name=name,allowed_dtypes=dtypes[name])
                    frame=decode_tensor(raw,max_payload_bytes=remaining)
                    # Charge every received frame, even repeated identical digests.
                    held_bytes+=len(raw);held_payload+=len(frame.payload)
                    if held_bytes>policy.max_input_bytes or held_payload>policy.max_payload_bytes:
                        raise ProtocolError('Cumulative installed input budget exceeded')
                inputs[name]=frame
                del raw
            for frame in inputs.values(): validate_finite(frame)
            outputs=operation(task,inputs)
            if not isinstance(outputs,dict) or set(outputs)!=set(task['outputs']): raise ProtocolError('Output names differ')
            digests={}
            for name,raw in outputs.items():
                frame=decode_tensor(raw); validate_finite(frame)
                spec=task['outputs'][name]
                if frame.header['name']!=name or any(frame.header[k]!=spec[k] for k in ('shape','dtype')):
                    raise ProtocolError('Output contract differs')
                digest=hashlib.sha256(raw).hexdigest()
                self.client.put_artifact(digest,raw,task=task,output_name=name); digests[name]=digest
            self.client.request('POST','/v1/work/result',dict(task,outputs=digests))
            return True
        except RemoteError:
            raise
        except (ValueError,MemoryError,RuntimeError) as exc:
            code='OUT_OF_MEMORY' if isinstance(exc,MemoryError) else ('EXECUTION_FAILED' if isinstance(exc,RuntimeError) else 'INVALID_TENSOR')
            self.client.request('POST','/v1/work/error',dict(task,code=code))
            return False
        finally:
            stop.set(); heartbeat.join(timeout=self.client.timeout+1)
    def run(self,stop_event,*,poll_interval=1):
        while not stop_event.is_set():
            self.run_once()
            stop_event.wait(poll_interval)
