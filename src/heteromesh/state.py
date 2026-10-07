"""Transactional sequential-job ledger; no model execution or network fallback."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class StateError(ValueError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class StateStore:
    """One SQLite connection protected per instance; SQLite serializes other processes."""

    def __init__(self, path: str | Path, *, clock=time.time):
        self.clock = clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, digest TEXT NOT NULL,
                request TEXT NOT NULL, state TEXT NOT NULL, epoch INTEGER NOT NULL,
                cursor INTEGER NOT NULL, pause_requested INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, job TEXT NOT NULL, epoch INTEGER NOT NULL,
                position INTEGER NOT NULL, node TEXT NOT NULL, deadline REAL NOT NULL,
                task TEXT NOT NULL, state TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS checkpoints (
                job TEXT NOT NULL, step INTEGER NOT NULL, epoch INTEGER NOT NULL,
                cursor INTEGER NOT NULL, PRIMARY KEY(job, step));
        ''')
        with self._tx():
            self.db.execute("UPDATE attempts SET state='invalidated' WHERE state='leased'")

    @contextmanager
    def _tx(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield
            except BaseException:
                self.db.execute('ROLLBACK')
                raise
            else:
                self.db.execute('COMMIT')

    def close(self):
        self.db.close()

    @staticmethod
    def validate_request(request):
        if not isinstance(request, dict) or not all(_is_digest(request.get(k)) for k in ('manifest_digest', 'profile_digest')):
            raise StateError('INVALID_PLAN')
        tasks = request.get('tasks')
        if not isinstance(tasks, list) or not tasks or len(tasks) > 10000:
            raise StateError('INVALID_PLAN')
        previous = {}
        previous_step = -1
        for task in tasks:
            if not isinstance(task, dict): raise StateError('INVALID_PLAN')
            for key in ('task_id', 'fragment_id', 'node_id', 'operation'):
                if not isinstance(task.get(key), str) or not task[key] or len(task[key]) > 128:
                    raise StateError('INVALID_PLAN')
            step = task.get('step_index')
            if type(step) is not int or step < 0 or step < previous_step:
                raise StateError('INVALID_PLAN')
            if task['task_id'] in previous: raise StateError('INVALID_PLAN')
            if not isinstance(task.get('parameters'), dict) or not isinstance(task.get('outputs'), dict) or not task['outputs']:
                raise StateError('INVALID_PLAN')
            if not isinstance(task.get('inputs'), dict): raise StateError('INVALID_PLAN')
            for reference in task['inputs'].values():
                if _is_digest(reference): continue
                if not isinstance(reference, dict) or set(reference) != {'task_id', 'output'}:
                    raise StateError('INVALID_PLAN')
                if reference['task_id'] not in previous or reference['output'] not in previous[reference['task_id']]:
                    raise StateError('INVALID_PLAN')
            previous[task['task_id']] = task['outputs']
            previous_step = step
        # Reject non-JSON values before any database mutation.
        try: _json(request)
        except (ValueError, TypeError, UnicodeError) as exc: raise StateError('INVALID_PLAN') from exc

    def _job(self, job_id):
        row = self.db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None: raise StateError('NOT_FOUND')
        return row

    def get_job(self, job_id):
        with self.lock:
            row = self._job(job_id)
            return {'job_id': row['id'], 'state': row['state'], 'recovery_epoch': row['epoch'], 'cursor': row['cursor'], 'task_count': len(json.loads(row['request'])['tasks'])}

    def submit_job(self, idempotency_key, request):
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 256:
            raise StateError('INVALID_IDEMPOTENCY_KEY')
        self.validate_request(request)
        digest = _digest(request)
        with self._tx():
            existing = self.db.execute('SELECT * FROM jobs WHERE idem=?', (idempotency_key,)).fetchone()
            if existing:
                if existing['digest'] != digest: raise StateError('CONFLICT')
                return self.get_job(existing['id'])
            job_id = secrets.token_hex(16)
            self.db.execute("INSERT INTO jobs(id,idem,digest,request,state,epoch,cursor) VALUES(?,?,?,?, 'queued',0,0)", (job_id, idempotency_key, digest, _json(request)))
            return self.get_job(job_id)

    def _resolve_inputs(self, job, task):
        inputs = {}
        request = json.loads(job['request'])
        for name, reference in task['inputs'].items():
            if isinstance(reference, str):
                inputs[name] = reference
                continue
            position = next(i for i, item in enumerate(request['tasks']) if item['task_id'] == reference['task_id'])
            row = self.db.execute("SELECT result FROM attempts WHERE job=? AND position=? AND state='committed' AND epoch<=? ORDER BY epoch DESC LIMIT 1", (job['id'], position, job['epoch'])).fetchone()
            if row is None: raise StateError('CHECKPOINT_INVALID')
            inputs[name] = json.loads(row['result'])['outputs'][reference['output']]
        return inputs

    def lease(self, node_id, *, duration=1800):
        if type(duration) not in (int, float) or not 0 < duration <= 86400: raise StateError('INVALID_DEADLINE')
        with self._tx():
            for job in self.db.execute("SELECT * FROM jobs WHERE state IN ('queued','running','pausing','waiting_for_nodes') ORDER BY rowid").fetchall():
                task = json.loads(job['request'])['tasks'][job['cursor']]
                if task['node_id'] != node_id: continue
                existing = self.db.execute("SELECT * FROM attempts WHERE job=? AND epoch=? AND position=? AND state='leased'", (job['id'], job['epoch'], job['cursor'])).fetchone()
                if existing and existing['deadline'] > self.clock():
                    return json.loads(existing['task'])
                if existing:
                    self.db.execute("UPDATE attempts SET state='expired' WHERE id=?", (existing['id'],))
                count = self.db.execute('SELECT count(*) FROM attempts WHERE job=? AND epoch=? AND position=?', (job['id'], job['epoch'], job['cursor'])).fetchone()[0]
                if count >= 3:
                    self.db.execute("UPDATE jobs SET state='failed' WHERE id=?", (job['id'],))
                    continue
                inputs = self._resolve_inputs(job, task)
                envelope = dict(task, job_id=job['id'], recovery_epoch=job['epoch'], attempt_id=secrets.token_hex(16), manifest_digest=json.loads(job['request'])['manifest_digest'], input_digest=_digest(inputs), inputs=inputs, deadline=self.clock()+duration)
                self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,'leased',NULL)", (envelope['attempt_id'], job['id'], job['epoch'], job['cursor'], node_id, envelope['deadline'], _json(envelope)))
                self.db.execute("UPDATE jobs SET state=? WHERE id=?", ('pausing' if job['pause_requested'] else 'running', job['id']))
                return envelope
            return None

    def commit_result(self, node_id, envelope, outputs):
        with self._tx():
            attempt = self.db.execute('SELECT * FROM attempts WHERE id=?', (envelope.get('attempt_id'),)).fetchone()
            if attempt is None or attempt['node'] != node_id: raise StateError('INVALID_LEASE')
            original = json.loads(attempt['task'])
            for field in ('job_id', 'recovery_epoch', 'task_id', 'fragment_id', 'step_index', 'manifest_digest', 'input_digest'):
                if type(envelope.get(field)) is not type(original[field]) or envelope.get(field) != original[field]:
                    raise StateError('INVALID_LEASE')
            if not isinstance(outputs, dict) or set(outputs) != set(original['outputs']) or not all(_is_digest(d) for d in outputs.values()):
                raise StateError('INVALID_OUTPUT')
            if attempt['state'] == 'committed':
                receipt = json.loads(attempt['result'])
                if receipt['outputs'] != outputs: raise StateError('CONFLICT')
                return receipt
            job = self._job(attempt['job'])
            if attempt['state'] != 'leased' or attempt['deadline'] <= self.clock() or job['epoch'] != attempt['epoch'] or job['state'] not in ('running', 'pausing'):
                raise StateError('INVALID_LEASE')
            receipt = {'commit_id': secrets.token_hex(16), 'outputs': outputs}
            self.db.execute("UPDATE attempts SET state='committed',result=? WHERE id=?", (_json(receipt), attempt['id']))
            cursor = job['cursor']+1
            tasks = json.loads(job['request'])['tasks']
            boundary = cursor == len(tasks) or tasks[cursor]['step_index'] != original['step_index']
            state = 'succeeded' if cursor == len(tasks) else ('paused' if boundary and job['pause_requested'] else ('pausing' if job['pause_requested'] else 'running'))
            self.db.execute('UPDATE jobs SET cursor=?,state=? WHERE id=?', (cursor, state, job['id']))
            if boundary:
                self.db.execute('INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?)', (job['id'], original['step_index'], job['epoch'], cursor))
            return receipt

    def cancel(self, job_id):
        with self._tx():
            job = self._job(job_id)
            if job['state'] not in ('succeeded', 'cancelled', 'failed'):
                self.db.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (job_id,))
                self.db.execute("UPDATE attempts SET state='invalidated' WHERE job=? AND state='leased'", (job_id,))
            return self.get_job(job_id)

    def pause(self, job_id):
        with self._tx():
            job = self._job(job_id)
            if job['state'] in ('succeeded', 'cancelled', 'failed'): raise StateError('CONFLICT')
            tasks = json.loads(job['request'])['tasks']
            at_boundary = job['cursor'] == 0 or tasks[job['cursor']-1]['step_index'] != tasks[job['cursor']]['step_index']
            active = self.db.execute("SELECT 1 FROM attempts WHERE job=? AND epoch=? AND position=? AND state='leased'", (job_id, job['epoch'], job['cursor'])).fetchone()
            state = 'paused' if at_boundary and not active else 'pausing'
            if job['state'] == 'paused': state = 'paused'
            self.db.execute('UPDATE jobs SET pause_requested=1,state=? WHERE id=?', (state, job_id))
            return self.get_job(job_id)

    def resume(self, job_id):
        with self._tx():
            job = self._job(job_id)
            if job['state'] != 'paused': raise StateError('CONFLICT')
            self.db.execute("UPDATE jobs SET pause_requested=0,state='queued' WHERE id=?", (job_id,))
            return self.get_job(job_id)

    def restore(self, job_id, *, step_index):
        with self._tx():
            job = self._job(job_id)
            if job['state'] in ('succeeded', 'cancelled', 'failed'): raise StateError('CONFLICT')
            point = self.db.execute('SELECT * FROM checkpoints WHERE job=? AND step=?', (job_id, step_index)).fetchone()
            if point is None: raise StateError('CHECKPOINT_INVALID')
            self.db.execute("UPDATE attempts SET state='invalidated' WHERE job=? AND state='leased'", (job_id,))
            self.db.execute("UPDATE jobs SET epoch=epoch+1,cursor=?,state='queued',pause_requested=0 WHERE id=?", (point['cursor'], job_id))
            return self.get_job(job_id)

    def report_error(self, node_id, envelope, code):
        allowed = {'OUT_OF_MEMORY', 'UNSUPPORTED_OPERATOR', 'MANIFEST_MISMATCH', 'INVALID_TENSOR', 'NODE_UNAVAILABLE', 'DEADLINE_EXCEEDED', 'DISK_FULL'}
        if code not in allowed: raise StateError('INVALID_ERROR')
        with self._tx():
            attempt = self.db.execute('SELECT * FROM attempts WHERE id=?', (envelope.get('attempt_id'),)).fetchone()
            if attempt is None or attempt['node'] != node_id or attempt['state'] != 'leased' or attempt['deadline'] <= self.clock(): raise StateError('INVALID_LEASE')
            original = json.loads(attempt['task'])
            if any(type(envelope.get(k)) is not type(original[k]) or envelope.get(k) != original[k] for k in ('job_id', 'recovery_epoch', 'task_id', 'fragment_id', 'step_index', 'manifest_digest', 'input_digest')):
                raise StateError('INVALID_LEASE')
            job = self._job(attempt['job'])
            if job['epoch'] != attempt['epoch'] or job['state'] not in ('running', 'pausing'):
                raise StateError('INVALID_LEASE')
            self.db.execute("UPDATE attempts SET state='failed' WHERE id=?", (attempt['id'],))
            state = 'waiting_for_nodes' if code in ('NODE_UNAVAILABLE', 'DEADLINE_EXCEEDED') else 'failed'
            self.db.execute('UPDATE jobs SET state=? WHERE id=?', (state, job['id']))
            return self.get_job(job['id'])

    def invalidate_node(self, node_id):
        with self._tx():
            jobs = self.db.execute("SELECT DISTINCT job FROM attempts WHERE node=? AND state='leased'", (node_id,)).fetchall()
            self.db.execute("UPDATE attempts SET state='invalidated' WHERE node=? AND state='leased'", (node_id,))
            for job in jobs:
                self.db.execute("UPDATE jobs SET state='waiting_for_nodes' WHERE id=? AND state IN ('running','pausing')", (job['job'],))
