"""Private local H3 complete-state persistence; optional pinned Torch runtime.

Not a network endpoint. Caller supplies trusted job/profile identity. CPU state
materialization and CPU RNG only; never claims physical-device qualification.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import torch
from diffusers import MiniMaxH3Scheduler
from .h3_runtime import verify_upstream
from .h3_metadata import DIFFUSERS_COMMIT
from .protocol import canonical_json, canonical_digest, parse_json, encode_tensor, decode_tensor, MAX_PAYLOAD_BYTES

SCHEDULER_SOURCE='307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd'
DTYPES={'float32':torch.float32,'bfloat16':torch.bfloat16,'int64':torch.int64,'uint8':torch.uint8}
BINDING={'job_id','model_manifest_sha256','profile_sha256','layout_sha256','diffusers_commit','torch_version','backend_profile','recovery_epoch','evaluation_count','mode','geometry','source_artifacts_sha256'}
GEOMETRY={'patch_t','patch_h','patch_w','latent_channels','audio_latent_channels','latent_frames','latent_height','latent_width','audio_channels','audio_latent_frames','num_condition_video_rows','num_condition_audio_rows'}
LAYOUT=('position_ids','token_tags','video_indices','audio_indices','text_indices','row_times')
TENSORS={'video_latents','audio_latents','prompt_embeds','condition_video_rows','condition_audio_rows',*LAYOUT}
TOP=BINDING|{'schema_version','kind','boundary','next_step','artifacts','schedulers','rng'}


class CheckpointError(ValueError):pass


def require(value,message):
    if not value:raise CheckpointError(message)


def exact(obj,keys):
    require(type(obj) is dict and set(obj)==set(keys),'unknown or missing checkpoint fields')


def integer(value,minimum=0):
    require(type(value) is int and minimum<=value<2**63,'invalid checkpoint integer')


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid checkpoint hash')


def source_identity():
    from .h3_runtime import upstream_identity
    source=Path(inspect.getfile(MiniMaxH3Scheduler)).read_bytes()
    return dict(upstream_identity(),scheduler_source_sha256=hashlib.sha256(source).hexdigest(),
                scheduler_source_bytes=len(source),scheduler_source_crlf=source.count(b'\r\n'))


def runtime():
    verify_upstream()
    require(hashlib.sha256(Path(inspect.getfile(MiniMaxH3Scheduler)).read_bytes()).hexdigest()==SCHEDULER_SOURCE,'scheduler source mismatch')


def frame(value,limit):
    require(isinstance(value,torch.Tensor) and value.layout==torch.strided,'invalid checkpoint tensor')
    names={v:k for k,v in DTYPES.items()};require(value.dtype in names,'unsupported checkpoint tensor dtype')
    require(value.numel()*value.element_size()<=limit,'checkpoint tensor budget exceeded')
    if value.is_floating_point():require(bool(torch.isfinite(value).all()),'nonfinite checkpoint tensor')
    cpu=value.detach().to('cpu').contiguous()
    # Flatten before uint8 view: scalar Float tensors otherwise cannot be reinterpreted.
    payload=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    raw=encode_tensor('checkpoint',names[value.dtype],list(value.shape),payload,max_payload_bytes=limit)
    return {'sha256':hashlib.sha256(raw).hexdigest(),'dtype':names[value.dtype],'shape':list(value.shape),'byte_length':len(payload)},raw


def layout_digest(geometry,tensors):
    return canonical_digest({'geometry':geometry,'artifacts':{name:frame(tensors[name],MAX_PAYLOAD_BYTES)[0] for name in LAYOUT}})


def validate_binding(binding):
    exact(binding,BINDING)
    for name in ('model_manifest_sha256','profile_sha256','layout_sha256','source_artifacts_sha256'):digest(binding[name])
    for name in ('job_id','backend_profile'):require(type(binding[name]) is str and 0<len(binding[name])<=256,'invalid checkpoint identity')
    require(binding['diffusers_commit']==DIFFUSERS_COMMIT and binding['torch_version']==torch.__version__,'incompatible checkpoint runtime')
    integer(binding['recovery_epoch']);integer(binding['evaluation_count'],1)
    require(binding['mode'] in ('t2va','fl2va','ref2va'),'unsupported checkpoint mode')
    geo=binding['geometry'];exact(geo,GEOMETRY)
    for name,value in geo.items():integer(value,0 if name.startswith('num_condition_') else 1)
    for size,patch in [('latent_frames','patch_t'),('latent_height','patch_h'),('latent_width','patch_w')]:require(geo[size]%geo[patch]==0,'invalid patch geometry')


def check_tensor(tensor,dtype,shape):
    require(isinstance(tensor,torch.Tensor) and tensor.dtype==dtype and tuple(tensor.shape)==tuple(shape),'checkpoint tensor shape/dtype mismatch')
    if tensor.is_floating_point():require(bool(torch.isfinite(tensor).all()),'nonfinite checkpoint tensor')


def validate_tensors(binding,tensors,schedulers):
    exact(tensors,TENSORS);g=binding['geometry'];e=binding['evaluation_count']
    cv,ca=g['num_condition_video_rows'],g['num_condition_audio_rows']
    vd=g['latent_channels']*g['patch_t']*g['patch_h']*g['patch_w'];ad=g['audio_latent_channels']
    v=cv+(g['latent_frames']//g['patch_t'])*(g['latent_height']//g['patch_h'])*(g['latent_width']//g['patch_w'])
    a=ca+g['audio_channels']*g['audio_latent_frames']
    p=tensors['prompt_embeds'];require(isinstance(p,torch.Tensor) and p.ndim==3 and p.shape[0]==1 and p.shape[1]>0 and p.shape[2]>0,'invalid prompt shape')
    t=p.shape[1];s=v+a+t
    shapes={'video_latents':(torch.float32,(v,vd)),'audio_latents':(torch.float32,(a,ad)),'condition_video_rows':(torch.float32,(cv,vd)),'condition_audio_rows':(torch.float32,(ca,ad)),'prompt_embeds':(torch.bfloat16,p.shape),'position_ids':(torch.float32,(s,3)),'token_tags':(torch.int64,(s,)),'video_indices':(torch.int64,(v,)),'audio_indices':(torch.int64,(a,)),'text_indices':(torch.int64,(t,)),'row_times':(torch.float32,(e,s))}
    for name,(dtype,shape) in shapes.items():check_tensor(tensors[name],dtype,shape)
    for modality,count in [('video',cv),('audio',ca)]:require(torch.equal(tensors[f'{modality}_latents'][:count].cpu().contiguous().view(torch.uint8),tensors[f'condition_{modality}_rows'].cpu().contiguous().view(torch.uint8)),'condition anchors changed')
    indices=[tensors[f'{name}_indices'].cpu() for name in ('video','text','audio')]
    require(torch.equal(torch.cat(indices).sort().values,torch.arange(s)),'layout is not exact disjoint cover')
    tags=tensors['token_tags'].cpu()
    for tag,ix in enumerate(indices):require(bool((tags[ix]==tag).all()),'layout tag mismatch')
    require(layout_digest(g,tensors)==binding['layout_sha256'],'layout digest mismatch')
    times=tensors['row_times'].cpu();vt=schedulers['video'].timesteps.cpu();at=schedulers['audio'].timesteps.cpu()
    vi,ti,ai=indices
    require(torch.equal(times[:,vi[cv:]],vt[:,None].expand(e,v-cv)) and torch.equal(times[:,ti],vt[:,None].expand(e,t)),'video/text row times mismatch')
    require(torch.equal(times[:,ai[ca:]],at[:,None].expand(e,a-ca)) and bool((times[:,ai[:ca]]==1).all()),'audio row times mismatch')
    if cv:
        anchors=times[0,vi[:cv]]
        require(bool(((anchors>=0)&(anchors<=1)).all()) and torch.equal(times[:,vi[:cv]],torch.maximum(vt[:,None],anchors[None,:])),'condition video row times mismatch')


class H3CheckpointStore:
    def __init__(self,root,*,max_artifact_bytes=MAX_PAYLOAD_BYTES,max_total_bytes=2*1024**3,durability_root=None):
        self.root=Path(root).absolute()
        bound=Path(durability_root).absolute() if durability_root is not None else Path(self.root.anchor)
        require(bound==self.root or bound in self.root.parents,'invalid durability root')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if os.name=='posix':self.root.chmod(0o700)
        for name in ('blobs','manifests'):(self.root/name).mkdir(exist_ok=True,mode=0o700)
        # Reconfirm directory entries on retries too: existence does not prove an
        # earlier mkdir's parent fsync succeeded. Default through filesystem
        # root, including a coordinator root which was itself newly created.
        # A caller may explicitly bound this at a pre-established durable root.
        directory=self.root
        while True:
            self._sync_dir(directory)
            if directory==bound:break
            directory=directory.parent
        integer(max_artifact_bytes,1);integer(max_total_bytes,1)
        self.limit=max_artifact_bytes;self.total_limit=max_total_bytes;self._fault=lambda stage:None
        self.durability='power-loss-fsync' if os.name=='posix' else 'process-crash-only'

    @contextmanager
    def _lock(self):
        db=sqlite3.connect(self.root/'publish-lock.sqlite',timeout=30)
        try:
            db.execute('BEGIN IMMEDIATE');yield
        finally:db.rollback();db.close()

    def _sync_dir(self,path):
        if os.name=='posix':
            fd=os.open(path,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)

    def _write(self,path,raw):
        fd,temp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();self._fault('blob_before_fsync');self._fault(f'{path.parent.name}_before_fsync');os.fsync(stream.fileno());self._fault('blob_after_fsync');self._fault(f'{path.parent.name}_after_fsync')
            os.replace(temp,path);self._sync_dir(path.parent)
        finally:
            if os.path.exists(temp):os.unlink(temp)

    def _put(self,folder,raw):
        sha=hashlib.sha256(raw).hexdigest();path=self.root/folder/sha
        if path.exists():
            with path.open('r+b') as stream:
                # Windows CRT _commit requires a writable handle, even for unchanged bytes.
                require(stream.read()==raw,'existing checkpoint blob corrupted');os.fsync(stream.fileno())
            self._sync_dir(path.parent)
        else:self._write(path,raw)
        return sha

    def _latest(self, sha=None):
        pointer=self.root/'latest'
        if sha is None:
            if not pointer.exists():return None
            require(pointer.stat().st_size==64,'invalid checkpoint pointer')
            sha=pointer.read_text(encoding='ascii')
        digest(sha)
        path=self.root/'manifests'/sha
        require(path.stat().st_size<=1024*1024,'checkpoint manifest exceeds limit')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==sha,'checkpoint manifest hash mismatch')
        value=parse_json(raw,max_bytes=1024*1024);exact(value,TOP)
        require(raw==canonical_json(value),'noncanonical checkpoint manifest')
        require(value['schema_version']==1 and type(value['schema_version']) is int and value['kind']=='h3_complete_step' and value['boundary']=='before_step','invalid checkpoint schema')
        return value

    def _capture(self,binding,next_step,tensors,schedulers,generators):
        runtime();validate_binding(binding);integer(next_step)
        e=binding['evaluation_count'];require(next_step<=e,'invalid complete-step boundary')
        exact(schedulers,{'video','audio'})
        for scheduler in schedulers.values():
            require(type(scheduler) is MiniMaxH3Scheduler,'unsupported checkpoint scheduler')
            check_tensor(scheduler.sigmas,torch.float32,(e+1,));check_tensor(scheduler.timesteps,torch.float32,(e,))
            require(bool((scheduler.sigmas[1:]<scheduler.sigmas[:-1]).all()) and scheduler.sigmas[-1].item()==0 and scheduler.sigmas[0].item()<=1,'invalid sigma grid')
            require(torch.equal(scheduler.timesteps,1-scheduler.sigmas[:-1]),'scheduler timestep mismatch')
            require((scheduler.step_index is None if next_step==0 else type(scheduler.step_index) is int and scheduler.step_index==next_step) and scheduler.begin_index is None,'partial scheduler boundary')
            require(type(scheduler.num_inference_steps) is int and scheduler.num_inference_steps==e,'scheduler evaluation count mismatch')
        validate_tensors(binding,tensors,schedulers)
        blobs={};total=0
        def ref(value):
            nonlocal total
            reference,raw=frame(value,self.limit);total+=len(raw)
            require(total<=self.total_limit,'checkpoint aggregate budget exceeded')
            blobs[reference['sha256']]=raw;return reference
        result=dict(binding,schema_version=1,kind='h3_complete_step',boundary='before_step',next_step=next_step,artifacts={name:ref(tensors[name]) for name in sorted(TENSORS)},schedulers={},rng=[])
        for name,scheduler in schedulers.items():
            shift=float(scheduler.config.shift);require(math.isfinite(shift) and shift>0 and torch.tensor(shift,dtype=torch.float32).item()==shift and scheduler._shift==shift,'scheduler shift is not exact FP32')
            sr=ref(torch.tensor(shift,dtype=torch.float32))
            result['schedulers'][name]={'class':'MiniMaxH3Scheduler','config_sha256':canonical_digest({'class':'MiniMaxH3Scheduler','shift':sr,'source_sha256':SCHEDULER_SOURCE}),'shift':sr,'sigmas':ref(scheduler.sigmas),'timesteps':ref(scheduler.timesteps),'step_index':scheduler.step_index,'begin_index':None,'num_inference_steps':e}
        require(type(generators) is list and 1<=len(generators)<=2,'invalid checkpoint generator list')
        roles=[]
        for role,generator in generators:
            require(role in ('request','condition_posterior') and role not in roles,'invalid/duplicate generator role');roles.append(role)
            require(isinstance(generator,torch.Generator) and str(generator.device)=='cpu','unsupported generator backend')
            result['rng'].append({'role':role,'engine':'torch.Generator','device_type':'cpu','device_index':None,'initial_seed':str(generator.initial_seed()),'state':ref(generator.get_state()),'draw_order_version':'h3-source-v1'})
        require('request' in roles,'request generator missing')
        return result,blobs

    def _publish(self,document,blobs):
        for raw in blobs.values():self._put('blobs',raw);self._fault('blob_durable')
        raw=canonical_json(document);require(len(raw)<=1024*1024,'checkpoint manifest exceeds limit')
        sha=self._put('manifests',raw);self._fault('manifest_durable');self._fault('before_pointer')
        self._write(self.root/'latest',sha.encode('ascii'));self._fault('after_pointer');return sha

    def publish(self,binding,next_step,tensors,schedulers,generators):
        try:
            document,blobs=self._capture(binding,next_step,tensors,schedulers,generators)
            with self._lock():
                previous=self._latest()
                if previous is None:require(next_step==0,'first checkpoint must be step zero')
                else:
                    require({k:previous[k] for k in BINDING}==binding,'checkpoint binding/epoch mismatch')
                    require(next_step==previous['next_step']+1,'checkpoint must advance exactly one complete step')
                return self._publish(document,blobs)
        except (KeyError,TypeError,RuntimeError,ValueError) as exc:raise CheckpointError('invalid checkpoint state') from exc

    def _load_tensor(self,reference,used):
        exact(reference,{'sha256','dtype','shape','byte_length'});digest(reference['sha256']);integer(reference['byte_length'])
        require(reference['byte_length']<=self.limit,'checkpoint tensor budget exceeded')
        path=self.root/'blobs'/reference['sha256'];require(path.stat().st_size<=self.limit+65540,'checkpoint frame too large')
        used[0]+=path.stat().st_size;require(used[0]<=self.total_limit,'checkpoint aggregate budget exceeded')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==reference['sha256'],'checkpoint tensor hash mismatch')
        decoded=decode_tensor(raw,max_payload_bytes=self.limit)
        require(decoded.header['dtype']==reference['dtype'] and decoded.header['shape']==reference['shape'] and len(decoded.payload)==reference['byte_length'],'checkpoint tensor reference mismatch')
        require(reference['dtype'] in DTYPES,'unsupported checkpoint dtype')
        dtype=DTYPES[reference['dtype']]
        tensor=torch.frombuffer(bytearray(decoded.payload),dtype=dtype).clone() if decoded.payload else torch.empty(0,dtype=dtype)
        return tensor.reshape(reference['shape'])

    def _load(self,binding,sha):
        try:
            runtime();validate_binding(binding)
            document=self._latest(sha);require(document is not None,'no complete checkpoint')
            require({k:document[k] for k in BINDING}==binding,'checkpoint binding/epoch mismatch')
            exact(document['artifacts'],TENSORS);exact(document['schedulers'],{'video','audio'})
            used=[0];read=lambda ref:self._load_tensor(ref,used)
            tensors={name:read(ref) for name,ref in document['artifacts'].items()};schedulers={}
            for name,record in document['schedulers'].items():
                exact(record,{'class','config_sha256','shift','sigmas','timesteps','step_index','begin_index','num_inference_steps'})
                shift=read(record['shift']);check_tensor(shift,torch.float32,())
                scheduler=MiniMaxH3Scheduler(shift=shift.item());scheduler.set_timesteps(sigmas=read(record['sigmas']))
                require(torch.equal(scheduler.timesteps,read(record['timesteps'])),'restored scheduler drift')
                scheduler._step_index=record['step_index'];scheduler._begin_index=record['begin_index'];schedulers[name]=scheduler
            generators=[]
            require(type(document['rng']) is list,'invalid generator records')
            for record in document['rng']:
                exact(record,{'role','engine','device_type','device_index','initial_seed','state','draw_order_version'})
                require(record['device_type']=='cpu' and record['device_index'] is None and record['engine']=='torch.Generator','unsupported generator backend')
                require(type(record['initial_seed']) is str and re.fullmatch('0|[1-9][0-9]{0,19}',record['initial_seed']) is not None and int(record['initial_seed'])<2**64,'invalid RNG seed')
                state=read(record['state']);require(state.dtype==torch.uint8 and state.ndim==1,'invalid RNG state')
                generator=torch.Generator().manual_seed(int(record['initial_seed']));generator.set_state(state)
                generators.append((record['role'],generator))
            rebuilt,blobs=self._capture(binding,document['next_step'],tensors,schedulers,generators)
            require(canonical_json(rebuilt)==canonical_json(document),'checkpoint schema or state inconsistency')
            return {'binding':dict(binding),'next_step':document['next_step'],'tensors':tensors,'schedulers':schedulers,'generators':generators}
        except (OSError,KeyError,TypeError,RuntimeError,ValueError) as exc:raise CheckpointError('checkpoint restore failed') from exc


    def prepare(self,binding,next_step,tensors,schedulers,generators):
        """Durable immutable state only; no authoritative pointer/epoch mutation."""
        document,blobs=self._capture(binding,next_step,tensors,schedulers,generators)
        with self._lock():
            for raw in blobs.values():self._put('blobs',raw);self._fault('blob_durable')
            raw=canonical_json(document);require(len(raw)<=1024*1024,'checkpoint manifest exceeds limit')
            sha=self._put('manifests',raw);self._fault('manifest_durable')
            return sha

    def restore(self,binding,*,new_epoch):
        try:
            integer(new_epoch);require(new_epoch>binding['recovery_epoch'],'recovery must advance epoch')
            with self._lock():
                document=self._latest();require(document is not None,'no complete checkpoint')
                sha=canonical_digest(document)
                restored=self._load(binding,sha)
                updated=dict(binding,recovery_epoch=new_epoch)
                rebuilt,blobs=self._capture(updated,restored['next_step'],restored['tensors'],restored['schedulers'],restored['generators'])
                self._publish(rebuilt,blobs)
                restored['binding']=updated
                return restored
        except (OSError,KeyError,TypeError,RuntimeError,ValueError) as exc:raise CheckpointError('checkpoint restore failed') from exc

    def load(self,binding,sha):
        with self._lock():return self._load(binding,sha)
