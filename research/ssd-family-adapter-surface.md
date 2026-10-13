# Research: draft-family adapter surface — one SSD core with thin adapters?

Ticket: PhillipChaffee/sglang#4 (map PhillipChaffee/sglang#1).
Question: is SSD implementable as ONE core with per-family draft adapters, and how thick is
each adapter? Owner's early personal priority: DFLASH (answered explicitly in §4).

Sources:
- SSD paper: arXiv 2603.03251v3 (Kumar, Dao, May; ICLR 2026), fetched 2026-10-12 —
  https://arxiv.org/abs/2603.03251 (Abstract, §3.1 Algorithm 1, §4.1.1, §4.3, Appendix E).
- SGLang fork: PhillipChaffee/sglang @ 45418d7dc6 (`main`). All `file:line` are exact there.

## 1. What the SSD core needs from a draft provider

From the paper (Algorithm 1, §3.1, §4.1.1, §4.3), the speculator runs as a **separate process
on separate hardware** ("in SSD the draft model is located on distinct hardware from the
target", §3.1) and, while verification of round T is ongoing, it must provide:

- **(a) Predicted-outcome fan-out inputs.** "Saguaro takes the top-F_k tokens at each
  lookahead position k in the draft logits, and adds those to the speculation cache. These are
  the top-F_k logits **excluding the sampled token** sent for verification at that position"
  (§4.1.1). The core needs per-position draft **logits/distributions** at the K+1 lookahead
  positions (not just argmax tokens), from which it forms outcomes v = (k, t*) and
  pre-speculates.
- **(b) Glue decode.** Each outcome absorb — append [k accepted tokens + hypothesized bonus
  t*] to the draft's context so the next chain rolls from it. This is the "speculate for
  outcomes" work; the REAL verify_outcome arrives later.
- **(c) Own draft KV cache** on the draft device, so hypothesized branches don't corrupt the
  committed chain.
- **(d) An async-able runner.** Outcome hit → instant serve of the pre-built chain; miss →
  fallback (in the paper: primary==backup just-in-time at small batch; at batch ≥ b* a very
  fast speculator — §4.3.1 footnote names n-gram-style non-neural speculators as the backup).

Appendix E pins the EAGLE caveat exactly: "since the SSD draft begins speculating round T+1
pre-emptively, it does not yet have target activations from round T (unlike standard EAGLE-3,
which waits for them). With lookahead K, the draft thus conditions on up to 2K self-generated
tokens, lowering acceptance on the latter K."

## 2. In-tree evidence base

### 2.1 Extension surface (what an SSD plugin must satisfy)

- `python/sglang/srt/speculative/spec_info.py:32-47` — `SpeculativeAlgorithm` enum
  (DFLASH, UNO, DSPARK, EAGLE, EAGLE3, FROZEN_KV_MTP, STANDALONE, NGRAM, NONE).
- `spec_info.py:67-91` — `SpeculativeAlgorithm.register(name, supports_overlap=…,
  validate_server_args=…, spec_class=…)`: plugin entry point; a `CustomSpecAlgo` subclass can
  override any `is_*()/supports_*/create_worker`.
- `spec_info.py:302-361` — `create_worker`: per-algorithm dispatch to the V2 worker class.
- `spec_registry.py:199-231` — `_assert_custom_spec_algo_conforms`: every `is_*`/`supports_*`
  method on the enum must exist on the plugin's `spec_class` (auto-checked at registration).
- `spec_info.py:364-444` — `SpecInputType` (EAGLE_DRAFT/…, DFLASH_DRAFT/VERIFY at :372-373,
  UNO_*, NGRAM_VERIFY) and the `SpecInput` ABC (:384) with `is_draft_input()/is_verify_input()`
  (:421-437), `custom_mask`, `ragged_verify_layout`, `num_tokens_per_req`.
- `spec_registry.py:26-175` — `CustomSpecAlgo` defaults: `has_draft_kv()` conservative True
  (:114-116); `supports_overlap=False` runs synchronously with a deprecation warning (:137-151).
- `spec_registry.py:183` — `NEXTN` reserved alias; `spec_info.py:60-61` resolves it to EAGLE.
- Cross-process ammo already present: `decoupled_spec_io.py:8-31,44-124,151-250,358-384`
  defines the verifier↔drafter message vocabulary (`DraftControlBatch`, `VerifyCommit`,
  `DraftClose`, `DraftTailStreamOutputBatch`, `DraftControlInbox`, `DraftSync` — "the verifier
  is the source of truth for committed tokens"); roles `verifier`/`drafter` and endpoints in
  `arg_groups/fields/disagg.py:46-56`; IPC wiring in `server_args.py:726,758-774`
  (`decoupled_spec_ipc_config` built in `PortArgs.init_new`). **Gap:** no code outside
  `decoupled_spec_io.py` consumes `DraftControlInbox`/`DraftTailStreamOutputBatch` (rg over
  the whole repo) — the drafter-side scheduler is scaffolded but not driven. That is the ONE
  clearly-missing piece of the paper's "[d] async runner" in-tree.
- Precedent for preparing next round's chain before the next iteration:
  `eagle_worker_v2.py:1441-1478` — under PP, EAGLE drafts the next round's chain immediately
  after verify ("drafting cannot wait for the next iteration") and relays the tree
  (`batch_output.next_verify_chain`). SSD's overlap of chain preparation with verification is
  a bigger version of this same seam.

### 2.2 What each worker looks like today

**STANDALONE** — `standalone_worker_v2.py:125` (`StandaloneWorkerV2(EAGLEWorkerV2)`); draft =
`StandaloneDraftWorker(EagleDraftWorker)` at :31-122 wrapping a full `TpModelWorker` with its
own embeddings and lm_head (`init_lm_head` overridden NOT to share target vocab modules,
:118-122). Vocab-equality gate :172-202. It reuses the whole EAGLE V2 machine, so its spec
inputs are `EagleDraftInput`/`EagleVerifyInput` where `hidden_states` is documented as "None
when the spec algorithm's draft doesn't read hidden_states (e.g., STANDALONE — vanilla LLM
draft)" — `eagle_info.py:163-166` (idle ctor passes `hidden_size=None`, :203-228).
Per-position draft distributions already exist: chain rollout collects `draft_probs`
(`(bs, num_steps, vocab)` under rejection sampling; `eagle_info.py:33-35,160-162`; built at
`eagle_worker_v2.py:727-733` via `sample_draft_proposal`/`renorm_draft_probs`, else
`fast_topk(probs, self.topk)` at :1029,1215).

**EAGLE/NEXTN** — `eagle_worker_v2.py`: round = `draft` (:614-694; sequential per-step
`draft_forward` at :696+, top-`topk` tree expansion via `select_top_k_tokens` at :766-771)
→ `verify` (:1907-1922 → `run_eagle_verify`) → `draft_extend` (:907;
`_draft_extend_for_decode` :1054-1239 consuming target hidden states captured during verify —
`EagleDraftExtendInput.hidden_states` "Target-model hidden states for the draft-extend
forward", `eagle_info.py:318-321`). Draft conditions on the target's LAST hidden state each
step (`spec_info.hidden_states` threaded through `draft_forward`, :703-786). Multi-layer
variant `multi_layer_eagle_worker_v2.py:117+` keeps per-step runners and the same
hidden-state conditioning (`chain_mtp_hidden_states`).

**DFLASH** — `dflash_worker_v2.py:370` (`DFlashWorkerV2`, 2942 lines) + `dflash_utils.py`
(1289 lines) + `dflash_info.py`/`dflash_info_v2.py`. Round in `forward_batch_generation`
(:2287-2942):
1. *Draft block*: `block_ids` = [bonus token, MASK…] (:2460-2529); embed via the TARGET's
   embedding table (+noise scale) (:2520-2530); ONE whole-block draft forward on the small
   draft model in TARGET_VERIFY mode (:2587-2635). Draft context comes exclusively from
   **target hidden states at target_layer_ids** — `build_target_layer_ids` picks mid-network
   capture points (`dflash_utils.py:366-403`; `DFlashDraftConfig.target_layer_ids`,
   `dflash_utils.py:537-603`); context rows are projected (`project_target_hidden`, :1880)
   and **kv_proj'd straight into the draft KV pool** via
   `_append_target_hidden_to_draft_kv_by_loc` (:1772-1958; fused Triton path
   `_append_target_hidden_fused` :1991-2034; prompt-token caller :2356-2361).
2. *Tokens*: block positions 1: sampled — greedily by the **borrowed target lm_head**
   (`_DflashDraftSampler`: "DFLASH's draft has no head of its own; it borrows the target
   `lm_head`", :127-137; eager fallback `_greedy_sample_from_vocab_parallel_head` :1550-1765),
   or selector/lilicorr heads fed by the same target lm_head (:836-864), or Domino GRU
   rollout (:871-902).
3. *Verify*: `DFlashVerifyInput` TARGET_VERIFY on the target (linear chain, `topk` forced 1,
   `dflash_info.py:27-43`); acceptance via chain rule (greedy
   `compute_dflash_correct_drafts_and_bonus` `dflash_utils.py:892-945`; sampled
   `compute_dflash_sampling_correct_drafts_and_bonus` `dflash_utils.py:990-1132`; selector-q
   via `accept_sampling` scatter `_selector_sampling_accept` :1471-1512).
4. *Glue — absorbed into verify, no extra draft pass*: the verify forward runs with
   `capture_hidden_mode=CaptureHiddenMode.FULL` (:2776) so per-block-position hidden states
   come back with the logits; :2903-2920 immediately re-projects them into the draft KV window
   at the same slots verify wrote (`cache_loc = self._draft_write_ids(kv_loc_plan)`, :2913),
   with prefix-valid commit (`set_kv_buffer_prefix_valid` :1907-1935) so only the accepted
   prefix length `commit_lens` becomes live next-round context. The draft never runs its own
   extend forward — EAGLE's glue pass **does not exist** in DFLASH. `_commit_accept`
   intercrops the bonus into the shifted candidate rows (:204-211), so the committed prefix
   includes a proxy row for the bonus token.
- Own KV pool: yes — separate draft runner + private compact sliding-window req->token;
  per-token byte budget split target+draft (`dflash_utils.py:151-205`,
  `dflash_draft_cell_size_per_token` / `scale_kv_cell_size_per_token_for_dflash`); shared
  allocator by default or private compact mapping (`alloc_memory_pool` :614-631; window
  rebuild :2549-2621).
- Spec inputs: DFLASH_DRAFT/DFLASH_VERIFY tags (`dflash_info_v2.py:60-68`,
  `dflash_info.py:60-61`); FutureMap relay under overlap; grouped with DSPARK as
  `is_dflash_family()` (`spec_info.py:126-127`).

**DSPARK** — `dspark_components/` (~7k lines incl. observability).
- Draft: `DraftBlockProposer.propose` + `draft_markov_block` (`dspark_draft.py:242-280,
  319-446`) — markov-head block drafting that, when not greedy, **emits full per-position step
  logits** (`logits_out` `[bs, gamma, vocab-width rows]`, `dspark_draft.py:251-257`);
  `DraftProposal` carries tokens, step logits, corrected logits (:396-446). Fan-out inputs (a)
  exist in the strongest form of any family.
- Glue/KV: `TargetHiddenKvInjector.inject_target_hidden` (`dspark_kv_inject.py:15-60`) — same
  absorbed-into-verify pattern as DFLASH.
- Budget planners: `DSparkVerifyPlanner`/`HostConfidenceBudgetPlanner`
  (`dspark_planner.py:69-1017`, `VerifyBudget`, `take_budget_decision` :234-243) implement
  per-round budgets — the same knob SSD calls cache budget B (§4.1). Per-request ragged verify
  lengths ride `RaggedVerifyLayout` (`spec_info.py:159-163`; `supports_ragged_verify` gates on
  `is_dspark()` only).
- Worker: `dspark_worker_v2.py:139+`, same `BaseSpecWorker` shape.

**FROZEN_KV_MTP** — `frozen_kv_mtp_worker_v2.py:91+`
(`FrozenKVMTPDraftWorker(EagleDraftWorkerBase, TpModelWorker)`). The frozen draft **"reads the
target KV cache read-only and owns no KV pool"** (module comment :16; `FrozenKVMTPContext` =
"Target KV pool + assistant-logical → target-physical layer map", `frozen_kv_mtp_info.py:27-40`,
bound via `bind_frozen_kv_context` :286-302). Reuses the EAGLE scheduler/attention contract:
`FrozenKVMTPDraftInput(EagleDraftInput)` / `FrozenKVMTPVerifyInput(EagleVerifyInput)`
(`frozen_kv_mtp_info.py:43-62`); scheduled through `is_eagle()` until promoted
(`spec_info.py:101-109`).

**NGRAM** — `ngram_worker.py:82`. No model: `draft_worker` returns None — "NGRAM has no draft
model; drafts come from the CPU-side corpus" (:151-154); drafts produced by the host trie
`self.ngram_corpus.batch_get` (:299-301), verified by the target (`NgramVerifyInput`,
`spec_info.py:374`). `has_draft_kv()` False (`spec_info.py:173-177`). No logits anywhere on
the draft side.

## 3. Per-family readiness against SSD core needs

Criteria: (a) per-position draft logits at K+1 lookahead positions (top-F, excl. sampled
token); (b) glue decode; (c) own KV cache on the draft device; (d) async-able runner.
"Adapter LOC" is a rough new/modified-lines magnitude for using the family as an SSD
**primary** on one SSD core. The core-side work (outcome fan-out scheduling, cache dict,
async runner over the decoupled seam) is charged to the SSD core, not the adapters.

| Family | (a) logits fan-out | (b) glue decode | (c) own KV | (d) async runner | Adapter LOC | Verdict |
|---|---|---|---|---|---|---|
| STANDALONE | Met — per-step topk probs already collected (`eagle_worker_v2.py:1029,1215`); widen topk or reuse draft_probs | Met — draft-extend consumes tokens only (`EagleDraftInput.hidden_states=None` path, `eagle_info.py:163-166`); hypothesized branches = draft forwards on hypothesized tokens | Met — private embeddings/head + own runner/KV (`standalone_worker_v2.py:71-122`) | Met — zero target-state dependency for round-T+1 speculation; fits the decoupled drafter as-is | ~200-400 | **THINNEST — do first (or first with DFLASH)** |
| DFLASH | Met w/ small change — sampler argmax-only today (`_DflashDraftSampler` :172-201); swap to per-row topk over the borrowed head; selector rows already topk-ish (`_selector_lattice` :229-241) | Met best-in-class — glue **absorbed into verify** (:2903-2920); hypothesized outcome (k,t*) ctx = slice of the already-materialized verify window (rows 0..k + proxy row); needs write-all-valid-rows instead of commit-lens-limited writes | Met — compact-window draft KV pool (`dflash_utils.py:151-205`; `alloc_memory_pool` :614-631) | Partial→Met — chain prep needs round-T verify hidden states (land at end of verify forward); per-outcome draft pass = one small block forward | ~300-600 | **THIN-MEDIUM — DFLASH verdict below** |
| EAGLE/NEXTN | Met — per-step `draft_probs (bs,steps,vocab)` machinery (`eagle_info.py:33-35,160-162`; `eagle_worker_v2.py:727-733`) | Partial — glue needs target hidden states from round T (`eagle_info.py:318-321`), so pre-emptive chains must switch to the paper's self-conditioning mode (App. E: ≤2K self-generated, deeper tokens drop); needs a self-recurrence draft-extend variant | Met — own draft runner + KV | Partial — target hidden must be shipped to drafter (`carries_draft_hidden_states`, `spec_info.py:179-182`); arrive near verify end | ~500-800 | MEDIUM |
| DSPARK | Met, richest — full per-position step logits when sampled (`dspark_draft.py:251-280`) | Met — same absorbed-in-verify injector (`dspark_kv_inject.py:15-60`) | Met | Same tie as DFLASH (target hidden) | ~600-900 | MEDIUM (planner/ragged machinery to un-couple) |
| FROZEN_KV_MTP | Met (EAGLE-shaped heads) | Met via EAGLE contract | **NOT met — owns no KV by design** (`frozen_kv_mtp_worker_v2.py:16`; `frozen_kv_mtp_info.py:27-40`) | Not without KV-ownership retrofit | ~800-1500 | THICK |
| NGRAM | **NOT met — no model, no logits** (`ngram_worker.py:151-154,299-301`) | N/A | None (`has_draft_kv()=False`) | Trivially async (host trie) | n/a | **OUT of scope as SSD primary; IN as fallback provider** (paper §4.3.1 footnote: n-gram speculators as backup) |

Ranking, thinnest → thickest: **STANDALONE → DFLASH → EAGLE/NEXTN → DSPARK → FROZEN_KV_MTP
(NGRAM: not a primary at all).** STANDALONE and DFLASH are close: STANDALONE is the most
target-decoupled; DFLASH has the cheapest per-outcome absorb (slices of an already-materialized
verify window) but cannot begin round-T+1 fan-out until the verify forward's hidden capture
lands. Once the fan-out device receives the verify hidden states, DFLASH's per-outcome pass is
a single tiny block forward, so the paper's prep-finishes-before-verification condition
(T_p < 1, Corollary 9) is easy to meet.

## 4. Explicit DFLASH verdict (owner priority)

DFLASH fits the one-core-with-adapter shape and is arguably the strongest per-round match to
SSD of all six families:

1. **Per-position draft logits (a):** the draft computes every block position's hidden rows in
   ONE parallel pass (mask-token rows; no intra-round self-conditioning, so the paper's EAGLE
   2K self-conditioning caveat does not apply within a round) and the borrowed target head
   converts rows → logits (`dflash_worker_v2.py:127-201`; eager :2744-2755). Today it only
   keeps the argmax (or selector top-K / lilicorr q with thresholds). Adapter work: emit
   top-F values+ids per row — same matmul, `topk` instead of `argmax`; ~50-100 LOC in the
   sampler plus graph-output plumbing (clone at the graph boundary; same pattern as
   `eagle_worker_v2.py:651-657`).
2. **Glue decode (b): absorbed into the verify forward.** No separate drafting glue pass
   exists (or is needed) in DFLASH: the verify forward captures FULL hidden states (:2776) and
   `_append_target_hidden_to_draft_kv_by_loc` projects+writes them into the draft KV window
   (:2903-2920, writer :1772-1958). For SSD, each hypothesized outcome (k, t*) needs exactly a
   sliced view of that materialized window: rows 0..k plus the proxy row for t* (the row that
   predicted the bonus — the window already establishes this proxy through `_commit_accept`
   :204-211 and prefix-valid slot writes :1907-1935). Adapter work: a write-all-valid-rows
   variant (drop the `commit_lens` prefix filter for hypothesized branches) + per-outcome
   `DFlashDraftInputV2` fragments. No new model forwards for absorb — the cheapest glue of any
   family.
3. **Own KV (c):** met — compact private draft pool with separate byte accounting
   (`dflash_utils.py:151-205`).
4. **Async runner (d):** the coupling is that round-T+1 ctx materialization consumes round-T
   verify hidden states, available at the end of the verify forward — fan-out drafting starts
   late but each per-outcome draft pass is a single block-shaped few-layer forward, and the
   family already overlap/PP-relays state (FutureMap). Full cross-process async needs the
   decoupled drafter driver (§6 open question 2) — shared by all families, not DFLASH-specific.
5. **Registration path:** `@SpeculativeAlgorithm.register("SSD_DFLASH",
   supports_overlap=True, spec_class=SsdDflashSpec)` returning `DFlashWorkerV2` (or subclass);
   the conformance check (`spec_registry.py:199-231`) enforces the `is_*()/supports_*()`
   surface; KV sizing via `has_draft_kv()` + `resolve_max_speculative_num_draft_tokens`
   (`spec_registry.py:121-132`). Spec-input types reuse DFLASH_DRAFT/DFLASH_VERIFY — the fan-out
   fan does not need new SpecInput shapes (§5).

**DFLASH verdict: GO — thin adapter (~300-600 LOC), no architectural blocker.**
Two caveats to scope at spec time: (1) the draft logits are a function of target ctx hidden
via the borrowed head — fine for the paper's outcome estimator (§4.1 prediction from "draft
logits" regardless of head provenance), but keep greedy-vs-sampled head variants
(`_DflashDraftSampler` / selector / lilicorr / Domino) consistent, or gate SSD mode to the
paths that expose per-position distributions; (2) chained outcomes longer than `block_size`
need multi-block continuation rollouts, which the draft does not express today (one block per
round) — cap K or extend the draft loop (§6 open question 3).

## 5. SSD core / adapter interface sketch

A `CustomSpecAlgo` plugin plus a thin per-family adapter object, reusing this fork's V2 worker
flows verbatim — the draft→verify→absorb loop stays; the SSD core sits *around* it, driving
per-outcome "speculate" on the draft device while the target scheduler verifies:

```python
# registered like any plugin: spec_info.py:67-91 + conformance spec_registry.py:199-231
@SpeculativeAlgorithm.register("SSD", supports_overlap=True, spec_class=SsdSpecAlgo)
def _factory(server_args): return _worker_for(server_args.bind_draft_family)
# first: STANDALONE | DFLASH | ...

class SsdDraftAdapter:                    # per family; sits ON TOP of the existing V2 worker
    def fanout_logits(self, draft_state, F_per_step) -> Fanout:
        # (a) per lookahead position k in [0..K]: top-F_k (ids, p) of the draft,
        # excluding the token already sampled for verification there (§4.1.1).
        # STANDALONE/EAGLE: draft_probs rows / fast_topk(F).
        # DFLASH: per-row topk over _DflashDraftSampler's logits (same matmul).
        # DSPARK: step_logits rows already in DraftProposal.
        # NGRAM: impossible — no logits (excluded from this interface).

    def build_chain(self, outcome: Outcome) -> SpecInput:
        # (b)+(c) "speculate_for_outcomes": absorb the hypothesized accept
        # (k accepted tokens + bonus t*), roll the next chain, return the
        # family's own verify SpecInput shape (EagleVerifyInput /
        # DFlashVerifyInput / ...) so the SSD core serves cache values as
        # ordinary verify inputs.
        # STANDALONE/EAGLE: draft forward on hypothesized tokens (own KV branch).
        # DFLASH/DSpark: slice the materialized verify ctx window (+ one small
        # draft block forward).

    def serve_cached(self, outcome) -> SpecInput   # hit -> instant hand-back
    def fallback_chain(self) -> SpecInput          # miss -> JIT / ngram backup (§4.3)

class SsdSpecAlgo(CustomSpecAlgo):        # the plugin shell
    # advertise + dispatch; family predicates fall through:
    #   supports_overlap=True (paper's parallel speculator)
    #   has_draft_kv() = adapter.has_kv
    #   carries_draft_hidden_states() = adapter needs hidden shipping
    #   create_worker() = SSD wrapper around the family's V2 worker class
```

The registry enforces the contract end-to-end: every `is_*()/supports_*()` predicate must
exist (`_assert_custom_spec_algo_conforms`, `spec_registry.py:199-231`); `SpecInput`
subclasses identify via `SpecInputType` and expose `is_draft_input()/is_verify_input()`
(`spec_info.py:364-437`); width bounds via `resolve_max_speculative_num_draft_tokens`
(`spec_registry.py:121-132`). Fan-out inputs ride existing spec-input types — no new shapes.

## 6. Confidence & open questions

Confidence: HIGH on the per-family facts (each cited file read this session); MEDIUM on the
adapter-LOC magnitudes (estimates, not measured); MEDIUM-HIGH on the ranking (only
EAGLE-vs-DSPARK is a judgement call, both "medium").

Open questions (none block triage; each would refine the spec):
1. **Per-outcome KV isolation for LM-draft families.** STANDALONE/EAGLE hypothesized absorbs
   need branch-local draft KV. Cheapest sketch: run hypothesized chains as extra virtual
   requests in the draft runner's own pool (draft-device scheduler bookkeeping, never seen by
   the target). Needs a small design spike.
2. **The decoupled drafter driver.** `decoupled_spec_io.py` transport + role args exist, but
   nothing constructs a drafter-side scheduler yet (no consumer of `DraftControlInbox`).
   Whichever family goes first, the SSD core's async runner must either drive that seam or
   start with in-process two-stream overlap; the paper's shape is one dedicated device per
   speculator.
3. **DFLASH chained outcomes > one block.** Outcome chains longer than `block_size` need
   continuation rollouts deeper than one block (the draft emits exactly one block per round).
   Decide: cap K at block size − 1 vs extend the draft loop to multi-block continuation.
4. **EAGLE self-conditioning mode.** A self-recurrence draft-extend (own hidden states instead
   of verify-captured) is a new draft-extend variant; scope it against
   `multi_layer_eagle_worker_v2.py`'s per-step runners, which already range over self-passes.
5. **NEXTN aliasing.** `NEXTN` is a reserved alias mapped to EAGLE (`spec_registry.py:183`,
   `spec_info.py:60-61`); an SSD plugin for NEXTN checkpoints must register its own spelling,
   not `NEXTN`.
