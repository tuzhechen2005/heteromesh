"""Trusted CPU snapshot adapter; the coordinator SQLite ledger is sole authority."""
import json
from pathlib import Path
import torch
from .state import StateError, _json
from .h3_checkpoints import H3CheckpointStore, BINDING, LAYOUT, CheckpointError, same_tensor_bytes
from .protocol import decode_tensor, canonical_json

class H3Recovery:
    def __init__(self,coordinator):
        self.app=coordinator;self.state=coordinator.state;self._stores={};self._fault=lambda stage:None
        self.active={}
        self.state.h3_commit=self.commit
        self.state.h3_lease=self.can_lease

    def can_lease(self,job):
        point=self.point(job['id'])
        active=self.active.get(job['id'])
        if point['ready'] and (active is None or active['digest']!=point['digest'] or active['binding']['recovery_epoch']!=job['epoch']):
            self.runtime_state(job['id']);active=self.active[job['id']]
        return active is not None and active['digest']==point['digest'] and active['binding']['recovery_epoch']==job['epoch']

    def refresh_current(self,job):
        if self.point(job)['ready']:self.can_lease(self.state._job(job))

    def runtime_state(self,job):
        with self.state.lock:
            point=self.point(job)
            # Re-read authority; failed SQL preparation cannot expose speculative runtime.
            state=self.store(job).load(json.loads(point['binding']),point['digest'])
            state['digest']=point['digest']
            self.active[job]=state
            return state

    def store(self,job):
        # job ids originate from the ledger, never user filesystem paths.
        # Restore preparation runs outside its SQL transaction. The shared
        # connection and lazy cache still require the ledger's thread lock.
        with self.state.lock:
            self.state._job(job)
            if job not in self._stores:self._stores[job]=H3CheckpointStore(self.app.root/'h3'/job)
            return self._stores[job]

    def point(self,job):
        with self.state.lock:row=self.state.db.execute('SELECT * FROM h3_boundaries WHERE job=?',(job,)).fetchone()
        if row is None:raise StateError('CHECKPOINT_REQUIRED')
        return row

    def initialize(self,job,binding,tensors,schedulers,generators):
        with self.state.lock:request=json.loads(self.state._job(job)['request']);tasks=request['tasks']
        if request.get('h3_recovery') is not True:raise StateError('INVALID_PLAN')
        if binding['job_id']!=job or binding['recovery_epoch']!=0 or binding['model_manifest_sha256']!=request['manifest_digest'] or binding['profile_sha256']!=request['profile_digest']:raise StateError('CHECKPOINT_INVALID')
        steps=sorted(set(t['step_index'] for t in tasks))
        if steps!=list(range(binding['evaluation_count'])):raise StateError('INVALID_PLAN')
        for i,task in enumerate(tasks):
            if i+1==len(tasks) or tasks[i+1]['step_index']!=task['step_index']:
                if set(task['outputs'])!={'video_latents','audio_latents'}:raise StateError('INVALID_PLAN')
                for name,spec in task['outputs'].items():
                    if spec!={'dtype':'float32','shape':list(tensors[name].shape)} and spec!={'dtype':'float32','shape':[1,*tensors[name].shape]}:raise StateError('INVALID_PLAN')
        sha=self.store(job).prepare(binding,0,tensors,schedulers,generators)
        self._fault('before_sql')
        with self.state._tx():
            row=self.state._job(job)
            if row['cursor']!=0 or row['epoch']!=0 or row['state'] not in ('queued','paused'):raise StateError('CONFLICT')
            existing=self.state.db.execute('SELECT * FROM h3_boundaries WHERE job=?',(job,)).fetchone()
            if existing:
                if existing['digest']!=sha:raise StateError('CONFLICT')
                return sha
            self.state.db.execute('INSERT INTO h3_boundaries VALUES(?,?,?,?,?,1)',(job,sha,_json(binding),0,0))
            self.active[job]=dict(self.store(job).load(binding,sha),digest=sha)
            self._fault('inside_sql')
        self._fault('after_sql');return sha

    def prepare(self,job,next_step,tensors,schedulers,generators):
        with self.state.lock:
            point=self.point(job);binding=json.loads(point['binding'])
        self._fault('prepare')
        return self.store(job).prepare(binding,next_step,tensors,schedulers,generators)

    def commit(self,job,attempt,outputs,sha,cursor):
        # Called only inside commit_result's write transaction, after current lease validation.
        if sha is None:raise StateError('CHECKPOINT_REQUIRED')
        point=self.point(job['id']);binding=json.loads(point['binding'])
        if not point['ready'] or binding['recovery_epoch']!=job['epoch']:raise StateError('CHECKPOINT_INVALID')
        try:
            new=self.store(job['id']).load(binding,sha)
            old=self.store(job['id']).load(binding,point['digest'])
            if new['next_step']!=old['next_step']+1 or new['next_step']!=json.loads(attempt['task'])['step_index']+1:raise StateError('CHECKPOINT_INVALID')
            for name in set(new['tensors'])-{'video_latents','audio_latents'}:
                if not same_tensor_bytes(new['tensors'][name],old['tensors'][name]):raise StateError('CHECKPOINT_INVALID')
            if any(not torch.equal(a.get_state(),b.get_state()) for (_,a),(_,b) in zip(new['generators'],old['generators'])):raise StateError('CHECKPOINT_INVALID')
            if [(role,gen.initial_seed()) for role,gen in new['generators']]!=[(role,gen.initial_seed()) for role,gen in old['generators']]:raise StateError('CHECKPOINT_INVALID')
            for name in ('video','audio'):
                a,b=new['schedulers'][name],old['schedulers'][name]
                if dict(a.config)!=dict(b.config) or not same_tensor_bytes(a.sigmas,b.sigmas) or not same_tensor_bytes(a.timesteps,b.timesteps):raise StateError('CHECKPOINT_INVALID')
            task=json.loads(attempt['task'])
            for name in ('video_latents','audio_latents'):
                grant=self.state.db.execute('SELECT 1 FROM output_grants WHERE attempt=? AND node=? AND job=? AND epoch=? AND name=? AND digest=?',(attempt['id'],attempt['node'],job['id'],job['epoch'],name,outputs[name])).fetchone()
                if grant is None:raise StateError('UNAUTHORIZED_OUTPUT')
                self.app._verify_artifact(outputs[name],task['outputs'][name],name)
                frame=decode_tensor((self.app.artifacts/outputs[name]).read_bytes())
                expected=new['tensors'][name]
                shape=frame.header['shape']
                if shape!=list(expected.shape) and shape!=[1,*expected.shape]:raise StateError('CHECKPOINT_INVALID')
                if frame.payload!=expected.contiguous().numpy().tobytes():raise StateError('CHECKPOINT_INVALID')
        except (CheckpointError,OSError,ValueError,KeyError) as exc:
            if isinstance(exc,StateError):raise
            raise StateError('CHECKPOINT_INVALID') from exc
        self._fault('before_sql')
        self.state.db.execute('UPDATE h3_boundaries SET digest=?,cursor=?,next_step=? WHERE job=?',(sha,cursor,new['next_step'],job['id']))
        new['digest']=sha
        self._fault('inside_sql')

    def before_commit(self):self._fault('before_sql_commit')

    def after_commit(self,job):
        self._fault('after_sql')
        self.runtime_state(job)
        # No latest cache exists. These are acknowledgment/response-loss boundaries.
        self._fault('before_response');self._fault('after_publication')

    def restore(self,job,*,request_id,expected_epoch):
        if not isinstance(request_id,str) or not 1<=len(request_id)<=128 or type(expected_epoch) is not int or expected_epoch<0:raise StateError('CHECKPOINT_INVALID')
        with self.state.lock:
            row=self.state._job(job);point=self.point(job)
            prior=self.state.db.execute('SELECT * FROM h3_restores WHERE job=? AND request=?',(job,request_id)).fetchone()
            if prior:
                if row['state'] in ('cancelled','failed'):raise StateError('CONFLICT')
                if prior['expected_epoch']!=expected_epoch or row['epoch']!=prior['new_epoch'] or point['digest']!=prior['digest'] or row['cursor']!=point['cursor']:raise StateError('CONFLICT')
                restored=self.store(job).load(json.loads(point['binding']),point['digest'])
                # A restart of the same committed recovery needs installation again, but no new epoch.
                with self.state._tx():
                    self.active[job]=dict(restored,digest=point['digest'])
                    self.state.db.execute('UPDATE h3_boundaries SET ready=1 WHERE job=?',(job,))
                return restored
            if row['epoch']!=expected_epoch or row['state'] in ('cancelled','failed'):raise StateError('CONFLICT')
            original_digest=point['digest'];binding=json.loads(point['binding'])
            restored=self.store(job).load(binding,original_digest)
        self._fault('restore_prepare')
        updated=dict(binding,recovery_epoch=expected_epoch+1)
        sha=self.store(job).prepare(updated,restored['next_step'],restored['tensors'],restored['schedulers'],restored['generators'])
        self._fault('restore_before_sql')
        with self.state._tx():
            row=self.state._job(job);current=self.point(job)
            concurrent=self.state.db.execute('SELECT * FROM h3_restores WHERE job=? AND request=?',(job,request_id)).fetchone()
            if concurrent:
                if concurrent['expected_epoch']!=expected_epoch or row['epoch']!=concurrent['new_epoch'] or current['digest']!=concurrent['digest'] or row['cursor']!=current['cursor'] or row['state'] in ('cancelled','failed'):raise StateError('CONFLICT')
                restored=self.store(job).load(json.loads(current['binding']),current['digest'])
                self.active[job]=dict(restored,digest=current['digest'])
                return restored
            if row['epoch']!=expected_epoch or current['digest']!=original_digest or row['state'] in ('cancelled','failed'):raise StateError('CONFLICT')
            self.state.db.execute("INSERT OR IGNORE INTO abandoned_commits SELECT id FROM attempts WHERE job=? AND position>=? AND state='committed'",(job,current['cursor']))
            self.state.db.execute("UPDATE attempts SET state='invalidated' WHERE job=? AND state='leased'",(job,))
            status='succeeded' if restored['next_step']==binding['evaluation_count'] else ('paused' if row['pause_requested'] else 'queued')
            self.state.db.execute('UPDATE jobs SET cursor=?,epoch=?,state=? WHERE id=?',(current['cursor'],expected_epoch+1,status,job))
            self.state.db.execute('UPDATE h3_boundaries SET digest=?,binding=?,ready=1 WHERE job=?',(sha,_json(updated),job))
            self.state.db.execute('INSERT INTO h3_restores VALUES(?,?,?,?,?)',(job,request_id,expected_epoch,expected_epoch+1,sha))
            restored['binding']=updated
            self._fault('restore_inside_sql')
        self._fault('restore_after_sql')
        with self.state.lock:
            current=self.point(job);live=self.state._job(job)
            if current['digest']!=sha or live['epoch']!=updated['recovery_epoch'] or live['cursor']!=current['cursor'] or live['state'] in ('cancelled','failed'):raise StateError('CONFLICT')
            self.active[job]=dict(restored,digest=sha)
        restored['binding']=updated;return restored
