---
status: accepted
---

# In-process-first topology and phase spine for SSD

The Speculative Speculative Decoding effort (wayfinder map) had to pick where the SSD core lands first: a paper-faithful split drafter on its own process/GPU, or an in-process drafter colocated with the target. We decided **in-process first**: phase 1 runs the SSD core plus the DFLASH draft family colocated on the target's TP4 GPUs with a blocking round loop — the pre-speculation cache, not the process split, is what supplies the asynchrony, so losslessness (the actual success bar) gets proven where ~2/3 of the machinery already exists (~800–1,300 LOC reused from spec-V2 vs ~4.2k–6k for the split path). The split-drafter seam is phase 2 behind a single deployment-mode config switch, so every later draft family enters with both topologies live and ships mode-parametrized parity regression; the decoupled-spec protocol's open questions (bulk tensor channel, wire types, ZMQ-under-load) become phase 2's tickets instead of blocking any usable SSD.

## Considered options

- **Split-drafter first (rejected):** paper-faithful hardware immediately, but it binds phase 1 to the 4.2k–6k LOC nine-piece transport build before one hypothesis is verified, on an unvalidated protocol (only the landed "1/N" schema exists, with zero consumers).
- **EAGLE first family (rejected for P1):** demand rank 1, but the most mode-carved: SSD needs a self-recurrence draft-extend variant to neutralize the paper's App. E self-conditioning caveat — the demand winner arrives at the thickest cost per runner.
- **STANDALONE first family (rejected for P1, reserved for P2):** thinnest adapter and a clean zero-target-state test bed, but ~no demand evidence, and its distinguishing property (instant fan-out usable on a detached draft device) only pays off once the split mode exists — exactly what phase 2 is bringing up.

## Consequences

- Phase 1 defers the paper's headline async-hardware story by one phase; accepted because the phase-1 success bar is lossless `--speculative-algorithm SSD`, not performance (no performance gate is locked).
- The in-process core must keep drafter-local state behind a seam abstraction so phase 2 can lift it out without adapter churn; family adapters stay topology-agnostic (the decoupled drafter driver belongs to the core, not the adapters).
- Phases after P2 stand on both modes live: family expansion (P3) and the parked support-matrix band (P4: PD disaggregation, DP attention, PP, grammar).
