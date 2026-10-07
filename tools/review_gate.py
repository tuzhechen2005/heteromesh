"""Validate actual, recorded non-author agent reviews. Never execute PR code."""

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone

MARKER = "<!-- heteromesh-agent-review-v1 -->"
AGENTS = {"/root", "/root/research", "/root/product", "/root/spec_qa"}
CONTEXT = "agent-review"


def parse_record(body):
    if not body.startswith(MARKER):
        raise ValueError("missing review marker")
    match = re.search(r"```json\s*\n(.*?)\n```", body, re.S)
    if not match:
        raise ValueError("missing JSON review")
    value = json.loads(match.group(1))
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("unsupported review version")
    if value.get("reviewer_agent") not in AGENTS:
        raise ValueError("unknown reviewer agent")
    if not re.fullmatch(r"[a-f0-9]{40}", value.get("head_sha", "")):
        raise ValueError("review must bind full SHA")
    if value.get("decision") not in {"approve", "changes_requested", "withdraw"}:
        raise ValueError("invalid decision")
    paths = value.get("reviewed_paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
        raise ValueError("missing reviewed scope")
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        raise ValueError("missing review evidence")
    return value


def evaluate(pr_body, sha, comments, permissions, invalidations=()):
    authors = re.findall(r"^Agent-Author:\s*(/root(?:/[a-z_]+)?)\s*$", pr_body or "", re.M)
    if len(authors) != 1 or authors[0] not in AGENTS:
        return False, "exactly one registered Agent-Author is required"
    latest = {}
    malformed = False
    for item in sorted(comments, key=lambda c: c["id"]):
        if not item.get("body", "").startswith(MARKER):
            continue
        if permissions.get(item["user"]["login"]) not in {"write", "maintain", "admin"}:
            continue
        try:
            review = parse_record(item["body"])
        except (ValueError, TypeError):
            malformed = True
            continue
        if review["head_sha"] != sha or review["reviewer_agent"] == authors[0]:
            continue
        if item["created_at"] != item["updated_at"]:
            continue
        latest[review["reviewer_agent"]] = (review, item["created_at"])
    if malformed:
        return False, "malformed review record from maintainer"
    for agent, after in invalidations:
        valid = [pair for reviewer, pair in latest.items()
                 if (agent == "*" or reviewer == agent) and pair[1] > after]
        if not valid:
            return False, "review edit/deletion requires a fresh review"
    if not latest:
        return False, "no current non-author review"
    if any(r["decision"] != "approve" for r, _ in latest.values()):
        return False, "unresolved request for changes or withdrawal"
    return True, "current SHA independently reviewed by another agent"


class GitHub:
    def __init__(self, repo, token):
        self.repo, self.token = repo, token

    def api(self, path, data=None):
        request = urllib.request.Request(
            "https://api.github.com/repos/" + self.repo + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Authorization": "Bearer " + self.token,
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def pages(self, path):
        values = []
        for page in range(1, 101):
            batch = self.api(path + f"?per_page=100&page={page}")
            values.extend(batch)
            if len(batch) < 100:
                return values
        raise RuntimeError("pagination limit; refusing partial review evidence")

    def status(self, sha, state, description):
        return self.api("/statuses/" + sha, {
            "state": state, "context": CONTEXT, "description": description[:140],
            "target_url": "https://github.com/" + self.repo + "/pulls",
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--pr", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    event_path = os.getenv("GITHUB_EVENT_PATH")
    with open(event_path, encoding="utf-8") if event_path else open(os.devnull) as handle:
        event = json.load(handle) if event_path else {}
    if "issue" in event and "pull_request" not in event["issue"]:
        return 0
    number = args.pr or event.get("pull_request", {}).get("number") or event.get("issue", {}).get("number")
    if not number or not args.repo:
        parser.error("repository and PR required")
    token = os.getenv("GITHUB_TOKEN") or subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    github = GitHub(args.repo, token)
    sha = None
    try:
        pr = github.api(f"/pulls/{number}")
        sha = pr["head"]["sha"]
        if pr["state"] != "open":
            return 0
        if not args.check_only:
            github.status(sha, "pending", "checking independent review evidence")
        changed = event.get("comment", {})
        if event.get("action") in {"edited", "deleted"} and (
            changed.get("body", "").startswith(MARKER)
            or event.get("changes", {}).get("body", {}).get("from", "").startswith(MARKER)
        ):
            try:
                agent = parse_record(changed["body"])["reviewer_agent"]
            except (ValueError, KeyError, TypeError):
                agent = "*"
            if not args.check_only:
                github.status(sha, "failure", "invalidation:" + agent)
            print("Review edited/deleted; append a fresh non-author review.")
            return 1
        comments = github.pages(f"/issues/{number}/comments")
        logins = {c["user"]["login"] for c in comments if c.get("body", "").startswith(MARKER)}
        permissions = {login: github.api(f"/collaborators/{login}/permission")["permission"] for login in logins}
        statuses = github.pages(f"/commits/{sha}/statuses")
        invalidations = [(s["description"].removeprefix("invalidation:"), s["created_at"])
                         for s in statuses if s["context"] == CONTEXT
                         and (s.get("description") or "").startswith("invalidation:")]
        passed, reason = evaluate(pr.get("body"), sha, comments, permissions, invalidations)
        if not args.check_only:
            github.status(sha, "success" if passed else "failure", reason)
        print(reason)
        return 0 if passed else 1
    except (urllib.error.URLError, ValueError, KeyError, RuntimeError) as error:
        # Do not print response bodies, tokens, or untrusted PR data.
        if sha and not args.check_only:
            github.status(sha, "failure", "review evidence read failed; fail closed")
        print("Review evidence unavailable:", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
