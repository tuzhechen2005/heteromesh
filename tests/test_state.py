import tempfile
import unittest
from pathlib import Path
from heteromesh.state import StateStore, StateError


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 1000
        self.db = Path(self.tmp.name) / 'ledger.db'
        self.store = StateStore(self.db, clock=lambda: self.now)
        self.request = {'manifest_digest': 'a'*64, 'profile_digest': 'b'*64, 'tasks': [
            {'task_id': 'a', 'fragment_id': 'first', 'step_index': 0, 'node_id': 'pc', 'operation': 'tiny', 'inputs': {'x': 'c'*64}, 'parameters': {}, 'outputs': {'y': {}}},
            {'task_id': 'b', 'fragment_id': 'last', 'step_index': 0, 'node_id': 'mac', 'operation': 'tiny', 'inputs': {'x': {'task_id': 'a', 'output': 'y'}}, 'parameters': {}, 'outputs': {'y': {}}},
            {'task_id': 'c', 'fragment_id': 'next', 'step_index': 1, 'node_id': 'pc', 'operation': 'tiny', 'inputs': {'x': {'task_id': 'b', 'output': 'y'}}, 'parameters': {}, 'outputs': {'y': {}}},
        ]}
    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()
    def submit(self):
        return self.store.submit_job('key', self.request)['job_id']
    def finish(self, node, task, digest='d'*64):
        return self.store.commit_result(node, task, {'y': digest})
    def test_submit_idempotent_conflict(self):
        job = self.submit()
        self.assertEqual(job, self.submit())
        changed = dict(self.request, profile_digest='f'*64)
        with self.assertRaises(StateError): self.store.submit_job('key', changed)
    def test_sequential_dependency_and_wrong_node(self):
        self.submit()
        self.assertIsNone(self.store.lease('mac'))
        first = self.store.lease('pc')
        with self.assertRaises(StateError): self.finish('mac', first)
        self.finish('pc', first)
        second = self.store.lease('mac')
        self.assertEqual(second['inputs'], {'x': 'd'*64})
    def test_duplicate_after_deadline_ack_and_conflict(self):
        self.submit()
        first = self.store.lease('pc')
        result = self.finish('pc', first)
        self.now += 4000
        self.assertEqual(result, self.finish('pc', first))
        with self.assertRaises(StateError): self.finish('pc', first, 'e'*64)
    def test_cancel_blocks_new_results(self):
        job = self.submit()
        task = self.store.lease('pc')
        self.store.cancel(job)
        with self.assertRaises(StateError): self.finish('pc', task)
        with self.assertRaises(StateError): self.store.resume(job)
    def test_pause_waits_complete_step_and_survives_restart(self):
        job = self.submit()
        first = self.store.lease('pc')
        self.store.pause(job)
        self.finish('pc', first)
        self.assertEqual(self.store.get_job(job)['state'], 'pausing')
        self.finish('mac', self.store.lease('mac'))
        self.assertEqual(self.store.get_job(job)['state'], 'paused')
        self.store.close()
        self.store = StateStore(self.db, clock=lambda: self.now)
        self.assertIsNone(self.store.lease('pc'))
        self.store.resume(job)
        self.assertEqual(self.store.lease('pc')['task_id'], 'c')
    def test_restart_invalidates_unsubmitted_attempt(self):
        self.submit()
        old = self.store.lease('pc')
        self.store.close()
        self.store = StateStore(self.db, clock=lambda: self.now)
        new = self.store.lease('pc')
        self.assertNotEqual(old['attempt_id'], new['attempt_id'])
        with self.assertRaises(StateError): self.finish('pc', old)
        self.finish('pc', new)
    def test_epoch_rollback_rejects_old_results(self):
        job = self.submit()
        self.finish('pc', self.store.lease('pc'))
        self.finish('mac', self.store.lease('mac'))
        old = self.store.lease('pc')
        self.store.restore(job, step_index=0)
        fresh = self.store.lease('pc')
        self.assertEqual(fresh['recovery_epoch'], 1)
        with self.assertRaises(StateError): self.finish('pc', old)
    def test_expired_lease_retries_then_fails(self):
        job = self.submit()
        for _ in range(3):
            self.assertIsNotNone(self.store.lease('pc'))
            self.now += 2000
        self.assertIsNone(self.store.lease('pc'))
        self.assertEqual(self.store.get_job(job)['state'], 'failed')
    def test_invalid_forward_reference_rejected(self):
        self.request['tasks'][0]['inputs'] = {'x': {'task_id': 'b', 'output': 'y'}}
        with self.assertRaises(StateError): self.submit()

    def test_failed_operation_is_terminal(self):
        job = self.submit()
        task = self.store.lease('pc')
        self.store.report_error('pc', task, 'UNSUPPORTED_OPERATOR')
        self.assertEqual(self.store.get_job(job)['state'], 'failed')
        self.assertIsNone(self.store.lease('pc'))
    def test_revocation_invalidates_lease_and_waits(self):
        job = self.submit()
        task = self.store.lease('pc')
        self.store.invalidate_node('pc')
        self.assertEqual(self.store.get_job(job)['state'], 'waiting_for_nodes')
        with self.assertRaises(StateError): self.finish('pc', task)
    def test_report_error_rejects_spoofed_node(self):
        self.submit()
        task = self.store.lease('pc')
        with self.assertRaises(StateError): self.store.report_error('mac', task, 'OUT_OF_MEMORY')

    def test_pause_at_existing_step_boundary_does_not_advance(self):
        job = self.submit()
        self.finish('pc', self.store.lease('pc'))
        self.finish('mac', self.store.lease('mac'))
        self.store.pause(job)
        self.assertEqual(self.store.get_job(job)['state'], 'paused')
        self.assertIsNone(self.store.lease('pc'))
    def test_result_rejects_identity_changes_and_boolean_epoch(self):
        self.submit()
        task = self.store.lease('pc')
        for change in ({'fragment_id':'fake'}, {'step_index':9}, {'recovery_epoch':False}):
            with self.subTest(change=change):
                with self.assertRaises(StateError): self.finish('pc',dict(task,**change))
    def test_error_rejects_identity_changes(self):
        self.submit()
        task = self.store.lease('pc')
        for change in ({'fragment_id':'fake'}, {'step_index':9}, {'recovery_epoch':False}):
            with self.subTest(change=change):
                with self.assertRaises(StateError): self.store.report_error('pc',dict(task,**change),'OUT_OF_MEMORY')
    def test_expired_error_cannot_fail_job(self):
        job = self.submit()
        task = self.store.lease('pc')
        self.now += 2000
        with self.assertRaises(StateError): self.store.report_error('pc',task,'OUT_OF_MEMORY')
        self.assertNotEqual(self.store.get_job(job)['state'],'failed')

    def test_wire_deadline_is_integer_seconds(self):
        self.now=1000.25
        self.submit()
        self.assertIs(type(self.store.lease('pc')['deadline']),int)

    def test_final_outputs_only_on_success_current_epoch(self):
        job=self.submit()
        self.assertNotIn('outputs',self.store.get_job(job))
        self.finish('pc',self.store.lease('pc'))
        self.finish('mac',self.store.lease('mac'))
        self.store.restore(job,step_index=0)
        self.finish('pc',self.store.lease('pc'),'e'*64)
        self.assertEqual(self.store.get_job(job)['outputs'],{'y':'e'*64})
