"""Pull worker executes only a caller-installed operation registry."""
import hashlib
import threading
from .client import RemoteError
from .protocol import decode_tensor, validate_finite, ProtocolError

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
            inputs={name:decode_tensor(self.client.get_artifact(digest,task)) for name,digest in task['inputs'].items()}
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
