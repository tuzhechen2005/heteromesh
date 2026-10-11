import tempfile
import unittest
from pathlib import Path
from heteromesh.state import StateStore, StateError

class RecoveryLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'ledger.db'
        self.store=StateStore(self.path)
        tasks=[]
        for i in range(5):
            tasks.append({'task_id':str(i),'fragment_id':str(i),'step_index':i,'node_id':'pc','operation':'tiny','parameters':{},'inputs':{'x':'a'*64 if i==0 else {'task_id':str(i-1),'output':'y'}},'outputs':{'y':{}}})
        self.job=self.store.submit_job('job',{'manifest_digest':'b'*64,'profile_digest':'c'*64,'tasks':tasks})['job_id']
    def tearDown(self):self.store.close();self.tmp.cleanup()
    def finish(self,digit):
        task=self.store.lease('pc');receipt=self.store.commit_result('pc',task,{'y':digit*64});return task,receipt
    def test_cannot_restore_descendant_checkpoint_after_rollback(self):
        self.finish('1');self.finish('2');self.finish('3')
        self.store.restore(self.job,step_index=0)
        with self.assertRaises(StateError):self.store.restore(self.job,step_index=2)
        self.assertEqual(self.store.get_job(self.job)['cursor'],1)
    def test_disconnected_restarted_branch_cannot_read_discarded_outputs(self):
        self.finish('1');old,receipt=self.finish('2');self.finish('3')
        self.store.restore(self.job,step_index=0)
        self.store.close();self.store=StateStore(self.path)
        lease=self.store.lease('pc')
        self.assertEqual(lease['task_id'],'1');self.assertEqual(lease['inputs']['x'],'1'*64)
        self.store.invalidate_node('pc')
        replacement=self.store.lease('pc')
        self.store.commit_result('pc',replacement,{'y':'e'*64})
        # Receipt retries from the abandoned epoch may be acknowledged but cannot advance.
        self.assertEqual(self.store.commit_result('pc',old,{'y':'2'*64}),receipt)
        next_task=self.store.lease('pc')
        self.assertEqual(next_task['task_id'],'2');self.assertEqual(next_task['inputs']['x'],'e'*64)
        # The obsolete future checkpoint must remain unusable after reconnect/recompute.
        with self.assertRaises(StateError):self.store.restore(self.job,step_index=2)
    def test_restore_rejects_boolean_step_identity(self):
        self.finish('1');self.finish('2')
        with self.assertRaises(StateError):self.store.restore(self.job,step_index=False)
    def test_abandoned_outputs_are_not_resolvable_even_if_active_reference_is_missing(self):
        self.finish('1');self.finish('2');self.finish('3')
        self.store.restore(self.job,step_index=0)
        self.finish('e')
        # Simulate loss/corruption of the current branch receipt, not an ordinary retry.
        self.store.db.execute("DELETE FROM attempts WHERE job=? AND epoch=1 AND state='committed'",(self.job,))
        with self.assertRaises(StateError):self.store.lease('pc')
