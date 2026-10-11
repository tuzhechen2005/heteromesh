# Recovery lineage audit and regression

Author: `/root/product`. Scope: REQ-007/010/011, protocol D-05. This remains a progress-only ledger; no H3 latent/scheduler/RNG checkpoint claim.

## Reproduced risk

Complete steps 0, 1 and 2 in a longer job, restore step 0, then select the still-stored step 2 checkpoint. The old implementation jumped the new epoch's cursor over work it had not recomputed. Input lookup also selected any latest historical committed result, even from the abandoned suffix.

Four RED regressions reproduced: restoring an obsolete descendant; doing so after disconnect/restart/recompute; a missing current-branch output falling back to abandoned old output; and accepting bool as step identity.

## Change

Restore accepts only an existing completed-step boundary at or behind the active cursor. In the same SQLite transaction it records suffix commits in an abandoned-commit table, deletes descendant checkpoints, invalidates pending leases and increments the recovery epoch. Dependencies exclude abandoned commits. Stored receipts remain available for identical retry acknowledgment, which cannot advance state. New descendants become available only after actual recomputation commits them.

The retained prefix remains usable across repeated rollbacks and coordinator restarts. A missing active-branch dependency fails CHECKPOINT_INVALID instead of silently borrowing output from an abandoned branch. Actual tensor-file integrity and H3 scheduler state remain separate service/model responsibilities.

## Verification

RED: four regressions failed against the prior ledger. GREEN: 18 existing state tests plus four lineage tests pass. State CI now discovers `test_state*.py`, so both suites run on all three operating systems; default pytest also includes them. Runtime/H3 physical-device evidence is unchanged.
