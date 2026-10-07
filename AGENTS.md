# HeteroMesh collaboration rules

Read `docs/development/workflow.md`, the relevant specification and test plan before changes. The product goal is genuine heterogeneous multi-device model inference, initially video generation, across Windows, macOS and iPhone. Never replace this with a single-machine application or claim mocked hardware evidence as physical-device validation.

- Use `codex/` branches and isolated worktrees for parallel code changes. Do not commit another agent's changes.
- Use SDD and meaningful RED/GREEN TDD evidence for behavior changes.
- Every PR records `Agent-Author: /root/<agent>` (root may use `/root`). A different agent must review the exact head SHA before merge.
- Use actual non-author code review, not a generated approval without reading the diff. Shared GitHub login cannot supply native independent-account approvals; do not misrepresent that limitation.
- Required CI and agent-review gates must pass on the latest commit. No administrative bypass of failed checks.
- Keep tokens, pairing secrets, IP addresses, device identifiers and private data out of public logs and fixtures.
- Fail clearly for unsupported runtimes/models and insufficient resources. Never silently execute a remote assignment on the coordinator.
- Never use pickle or remotely supplied executable code as a tensor/task protocol.
- Record simulated, local multi-process, physical-device and target-model evidence separately. iPhone/H3 validation stays NOT RUN until actually performed.
- Never download model weights or initiate long GPU jobs merely to run the default test suite.
- Keep protocol contracts and cross-language fixtures consistent. Numeric comparisons use explicit tolerances rather than assuming bit equality across hardware.
