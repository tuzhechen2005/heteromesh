import json
import unittest

from tools.review_gate import evaluate, parse_record

SHA = "a" * 40
BODY = "Agent-Author: /root\n"
MARKER = "<!-- heteromesh-agent-review-v1 -->"


def comment(agent="/root/research", decision="approve", sha=SHA, ident=1):
    record = {"version": 1, "reviewer_agent": agent, "head_sha": sha,
              "decision": decision, "reviewed_paths": ["docs/spec/protocol-v1.md"],
              "summary": "Read the diff and ran the relevant tests."}
    time = f"2026-10-06T12:00:{ident:02}Z"
    return {"id": ident, "body": MARKER + "\n```json\n" + json.dumps(record) + "\n```",
            "created_at": time, "updated_at": time, "user": {"login": "maintainer"}}


class ReviewTests(unittest.TestCase):
    def check(self, comments, body=BODY, invalidations=()):
        return evaluate(body, SHA, comments, {"maintainer": "admin"}, invalidations)

    def test_missing_review_and_self_review_fail(self):
        self.assertFalse(self.check([])[0])
        self.assertFalse(self.check([comment(agent="/root")])[0])

    def test_current_non_author_review_passes(self):
        self.assertTrue(self.check([comment()])[0])

    def test_stale_sha_cannot_approve(self):
        self.assertFalse(self.check([comment(sha="b" * 40)])[0])

    def test_blocking_review_overrides_another_approval(self):
        reviews = [comment(), comment("/root/product", "changes_requested", ident=2)]
        self.assertFalse(self.check(reviews)[0])
        reviews.append(comment("/root/product", "approve", ident=3))
        self.assertTrue(self.check(reviews)[0])

    def test_withdrawal_requires_same_reviewer_to_return(self):
        reviews = [comment(), comment(decision="withdraw", ident=2),
                   comment("/root/product", ident=3)]
        self.assertFalse(self.check(reviews)[0])

    def test_edited_comment_invalid(self):
        review = comment()
        review["updated_at"] = "2026-10-06T12:10:00Z"
        self.assertFalse(self.check([review])[0])

    def test_deleted_review_invalidation_survives_comment_list_refresh(self):
        inv = [("/root/research", "2026-10-06T12:00:02Z")]
        self.assertFalse(self.check([comment("/root/product", ident=3)], invalidations=inv)[0])
        self.assertTrue(self.check([comment(ident=4)], invalidations=inv)[0])

    def test_permissions_are_not_inferred_from_agent_string(self):
        self.assertFalse(evaluate(BODY, SHA, [comment()], {"maintainer": "read"})[0])

    def test_ambiguous_author_rejected(self):
        self.assertFalse(self.check([comment()], body=BODY + "Agent-Author: /root/product")[0])

    def test_unknown_agent_rejected(self):
        self.assertFalse(self.check([comment(agent="/root/invented")])[0])

    def test_malformed_record_rejected(self):
        review = comment()
        review["body"] = MARKER + '\n```json\n{"version":true}\n```'
        self.assertFalse(self.check([review])[0])

    def test_empty_review_scope_rejected(self):
        review = comment()
        record = parse_record(review["body"])
        record["reviewed_paths"] = []
        review["body"] = MARKER + "\n```json\n" + json.dumps(record) + "\n```"
        self.assertFalse(self.check([review])[0])


if __name__ == "__main__":
    unittest.main()
