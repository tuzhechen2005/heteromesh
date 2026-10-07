"""The required commit status gates merging; Actions reports evaluator health."""
import os
import sys
import unittest
from unittest.mock import patch
from tools import review_gate
from test_review_gate import SHA, BODY, comment


class FakeGitHub:
    def __init__(self, comments):
        self.comments=comments;self.statuses=[]
    def api(self,path,data=None):
        if path=='/pulls/9':return {'head':{'sha':SHA},'state':'open','body':BODY}
        if path.endswith('/permission'):return {'permission':'admin'}
        raise AssertionError(path)
    def pages(self,path):
        return self.comments if '/comments' in path else []
    def status(self,sha,state,description):self.statuses.append((sha,state,description))


class StatusTests(unittest.TestCase):
    def run_gate(self, comments, check=False):
        github=FakeGitHub(comments)
        args=['review_gate.py','--repo','example/repo','--pr','9']+(['--check-only'] if check else [])
        with patch.dict(os.environ,{'GITHUB_TOKEN':'test-token','GITHUB_EVENT_PATH':''},clear=True),patch.object(sys,'argv',args),patch.object(review_gate,'GitHub',return_value=github):
            code=review_gate.main()
        return code,github.statuses

    def test_no_review_is_pending_and_evaluator_succeeds(self):
        code,statuses=self.run_gate([])
        self.assertEqual(code,0)
        self.assertEqual(statuses[-1][1],'pending')

    def test_stale_review_does_not_become_success(self):
        code,statuses=self.run_gate([comment(sha='b'*40)])
        self.assertEqual(code,0)
        self.assertEqual(statuses[-1][1],'pending')

    def test_requested_changes_blocks_merge_without_failing_evaluator(self):
        code,statuses=self.run_gate([comment(decision='changes_requested')])
        self.assertEqual(code,0)
        self.assertEqual(statuses[-1][1],'failure')

    def test_valid_approval_is_success(self):
        code,statuses=self.run_gate([comment()])
        self.assertEqual(code,0)
        self.assertEqual(statuses[-1][1],'success')

    def test_check_only_still_returns_nonzero_for_unapproved_pr(self):
        code,statuses=self.run_gate([],check=True)
        self.assertEqual(code,1)
        self.assertEqual(statuses,[])

    def test_evidence_read_error_still_fails_actions_and_merge(self):
        github=FakeGitHub([])
        github.pages=lambda path: (_ for _ in ()).throw(ValueError('unavailable evidence'))
        with patch.dict(os.environ,{'GITHUB_TOKEN':'test-token','GITHUB_EVENT_PATH':''},clear=True),patch.object(sys,'argv',['gate','--repo','example/repo','--pr','9']),patch.object(review_gate,'GitHub',return_value=github):
            self.assertEqual(review_gate.main(),1)
        self.assertEqual(github.statuses[-1][1],'failure')
