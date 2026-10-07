"""Check requirement traceability and required design artifacts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["docs/product/PRD.md", "docs/research/feasibility.md",
            "docs/spec/system-spec.md", "docs/spec/protocol-v1.md",
            "docs/testing/test-plan.md", "docs/development/workflow.md",
            "docs/reviews/design-decision.md"]


def main():
    for path in REQUIRED:
        if not (ROOT / path).is_file():
            raise SystemExit("Missing design artifact: " + path)
    spec = (ROOT / "docs/spec/system-spec.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs/testing/test-plan.md").read_text(encoding="utf-8")
    requirements = set(re.findall(r"REQ-\d{3}", spec))
    covered = set(re.findall(r"REQ-\d{3}", plan))
    missing = requirements - covered
    if missing:
        raise SystemExit("Missing test-plan mapping: " + ", ".join(sorted(missing)))
    for path in ROOT.glob("docs/**/*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            candidate = path.parent / target.split("#", 1)[0]
            if not candidate.exists():
                raise SystemExit(f"Broken relative link in {path.relative_to(ROOT)}: {target}")
    print(f"Design artifacts present; {len(requirements)} requirements mapped. This is not hardware evidence.")


if __name__ == "__main__":
    main()
