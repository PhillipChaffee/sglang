# Reuse map: SSD hypothesis machinery vs SGLang's existing spec-v2 work

**Ticket:** PhillipChaffee/sglang#3 (map #1) · **Date:** 2026-10-12 · **Status:** research findings

## Question

How much of SSD's (Saguaro, arXiv 2603.03251 — "Speculative Speculative Decoding", ICLR 2026) hypothesis
machinery can reuse sglang's existing tree/fork/estimator work, and where is new machinery unavoidable?
SSD needs, per round: (1) glue decode — the drafter absorbs verified tokens plus prior speculation via an
extend pass; (2) fork F×(K+1) hypotheses; (3) K-step parallel branch decode on the draft device; (4) an
outcome-keyed branch cache served instantly on hit.

## Sources

- Paper: https://arxiv.org/abs/2603.03251 (v3). Abstract confirms the mechanism: while verification runs,
  the drafter "predicts likely verification outcomes and prepares speculations pre-emptively … If the actual
  verification outcome is then in the predicted set, a speculation can be returned immediately" — i.e. the
  *prediction mechanism is the fork + branch cache*, not a separate scorer model. Saguaro is "on average 30%
  faster than optimized speculative decoding baselines".
- Reference engine: https://github.com/tanishqkumar/ssd at HEAD `d7eb8fa0` — inspected
  `ssd/engine/draft_runner.py`, `ssd/engine/helpers/mask_helpers.py`, `ssd/utils/async_helpers/
  async_spec_helpers.py`, `ssd/engine/block_manager.py`, and directory trees.
- Local anchors: line numbers from this fork's working tree at `45418d7dc6`.

## Round anatomy of SSD's reference engine (what each piece actually does)

| Phase | Reference implementation | What it does |
|---|---|---|
| Cache serve / miss | `draft_runner.hit_cache_and_respond`, `_reset_tree_cache_tensors` | Round keyed `(seq_id, k_idx, rec_token)` tensor-backed lookup; vectorized membership (`all(eq, dim=2)` → `cache_hits`); served slots get tokens `[T,K]`, logits `[T,K+1,V]`, activations; miss → `jit_speculate` (K-step sequential drafter over the run). |
| Glue-decode input | `make_glue_decode_input_ids` (`async_spec_helpers.py`) | `[rec_token, spec_0..spec_{K-1}]` per seq → flat `B*(K+1)` payload. |
| Glue decode forward | `draft_runner._build_tree_batch` (EAGLE: `prepare_glue_decode_ctxt_eagle`) | One varlen decode-style forward: per-seq extend tokens (from the previous accepted prefix) + recovery token + K spec tokens, packed with `cu_seqlens_q = n_ext + (K+1)`; extracts per-position `(K+1)` logits and prenorms (hidden-state self-conditioning for EAGLE). |
| Fork | `get_forked_recovery_tokens_from_logits` (`async_spec_helpers.py`) | scatter `-inf` on already-sampled tokens at the first K positions, one `topk` at `k_max`, per-`(b, position)` fan pruning from hit vs miss lists → flat `[B, MQ_LEN]` fork tokens; hidden states via `torch.repeat_interleave(prenorms, per-batch fan)`. Row tables `_fan_idx_hit/_miss = arange(K+1).repeat_interleave(F)` (`draft_runner._init_prealloc_buffers`). |
| Branch decode | `draft_runner._decode_tree`, `_decode_tree_step`, `_compute_step_positions_and_slot_maps` | K sequential steps; each step all `N = B*MQ_LEN` rows decode in parallel, one token per row into per-row block-table slots (`dbt`), per-row `context_lens` (`step_context_lens = positions[:, -1] + 1`), `tree_decode_step=depth`; returns `spec_tokens [N,K]`, `spec_logits [N,K,V]`, `spec_activations [N,K,H]`. |
| Per-branch sparse masks | `helpers/mask_helpers.py` (`get_custom_mask` → cached / vectorized) | Per-row mask = prefix ones + tril(glue K+1) repeated per fan row (hit or miss list) + identity-diagonal blocks for each decoded step. Precomputed components, `flat_blocks_after_cat` for B>8. |
| Populate cache | `draft_runner._populate_tree_cache` | Whole-tensor store: keys `[N,3]`, tokens `[N,K]`, logits `[N,K,V]`, activations `[N,K,H]`. |
| KV allocation | `block_manager.py` (`may_append(..., lookahead=...)`, ref-counted fork blocks `*_deallocate_n_blocks`, separate `draft_block_table`) | Lookahead wedge = `compute_megaspec_lookahead() = K+1 + K*MQ_LEN` tokens per seq pre-reserved in the draft block table. |

Key structural fact: SSD's "tree" is **not** a deep k-ary proposal tree like EAGLE/UNO — it is a fixed-width
multi-query row set (`N = B*Σ(fan_per_position)`) whose only structural mask is the tril-glue lineage plus
per-step diagonal blocks. That difference drives most verdicts below.

## Reuse map

Verdicts: **REUSE** = direct call, no new code; **ADAPT** = reuse the primitive, extend it; **BUILD-NEW** = no
equivalent exists. "New LOC" counts only what is *new*.

### 1. Glue decode (absorb verified tokens + prior speculation via one extend pass) — **ADAPT**

- **Reused from sglang:** `EAGLEWorkerV2` draft-extend machinery — `eagle_worker_v2.py:1054
  _draft_extend_for_decode` (one extend over the accepted prefix every decode round), `:907 draft_extend`,
  captured in `eagle_draft_extend_cuda_graph_runner.py:95 EAGLEDraftExtendCudaGraphRunner`. Variable
  per-request extend lengths already exist: `ragged_verify.py:46 RaggedVerifyLayout`, `:287
  compute_uniform_extend_lengths`, `:307 compute_ragged_extend_lengths`, feeding verify prep via
  `eagle_utils.py:571-581 assign_extend_cache_locs_uniform_func`.
- **What differs:** SSD's glue token stream per request is *variable-length*: `n_ext[b] + (K+1)` packed into
  cu_seqlens rows, with a fused (ids, hidden-prenorm) interleave — extend tokens and the recovery token are
  target-conditioned (`fc(acts)`), spec tokens are self-conditioned (prenorm passthrough). sglang's uniform
  `(K+1)`-wide extend capture needs a varlen packed variant.
- **New LOC sketch:** ~120–200 (id/prenorm packing + cu_seqlens lookup; prototype's `_build_tree_batch`
  EAGLE path ≈ 80 lines, `prepare_glue_decode_ctxt_eagle` ≈ 40; CG-capture rebind ≈ 50).

### 2. Fork — spawn F×(K+1) hypotheses per sequence — **ADAPT**

- **Reused from sglang:** the topk-expansion skeleton is `spec_utils.py:422 select_top_k_tokens` (per-step
  candidate expansion with the i=0 special case `_select_top_k_tokens_first`), `spec_utils.py:520
  traverse_tree`, and the inter-iteration relay of fork inputs through FutureMap's spec extras —
  `overlap_utils.py:140 RelayPayload` / `:166 from_draft_input` publishes `bonus_tokens`, `topk_p`,
  `topk_index`, `hidden_states` to the next iteration.
- **What differs:** SSD forks in one pass at the K+1 lookahead positions over the K+1 glue logits, masking
  the already-sampled tokens (scatter −inf before a single topk, per-position fan counts → flat
  `[B, MQ_LEN]`), with per-batch hit-vs-miss fan lists (`fan_out_t` / `fan_out_t_miss`) selected by
  `cache_hits`. It also must fork *hidden states* (`repeat_interleave` of the prenorm) alongside tokens.
  sglang's `select_top_k_tokens` expands a *tree* (parents deepen each step); SSD fans a *canopy* (only
  depth-1 forks, then K chain steps below each fork).
- **New LOC sketch:** fork kernel + fan-index tables ≈ 80–120 (prototype topk ≈ 60 lines plus the
  `_init_prealloc_buffers` fan tables ≈ 25).

### 3. Per-branch sparse attention masks — **REUSE the mask ABI, BUILD-NEW the builder**

- **Reused from sglang:** `eagle_utils.py:144 TreeMaskMode` (FULL_MASK / QLEN_ONLY / QLEN_ONLY_BITPACKING)
  and its consumers: `eagle_utils.py:156 build_tree_kernel_efficient` (dispatching to
  `sgl_build_tree_kernel_efficient` CUDA, Triton `:301`, CPU, NPU variants) with buffer management at
  `eagle_utils.py:178-219`, and the caller plumbing in `eagle_worker_common.py:466-469` inside
  `build_eagle_verify_input` (:395), incl. preallocated `verify_mask` buffer handling (:428-433). The
  *target*-side spec verify attention consumes these masks today and is reused untouched for SSD's served
  chain.
- **What differs:** SSD's decode-side mask is *appended-only*: per row = prefix ones + tril(glue K+1)
  fan-repeated (hit or miss variant) + identity-diagonal blocks for each decoded step
  (`mask_helpers.get_custom_mask`, cached and vectorized variants; `flat_blocks_after_cat` for B>8). Row
  count is `B*MQ_LEN` with row→block-table mapping; EAGLE's kernel emits `retrieve_index`/
  `retrieve_next_token`/`retrieve_next_sibling` linkage over the depth-limited candidate ABI
  (`uno_tree.py:24-48` documents that ABI) that SSD's flat canopy does not need.
- **New LOC sketch:** builder ≈ 60–150 (prototype: `_get_custom_mask_optimized` + cached-component
  precompute ≈ 120, vectorized variant ≈ 130; a single merged torch/triton builder for sglang lands
  mid-bound). Backend plan wiring is the real cost — see row 4.

### 4. K-step parallel branch decode on the draft device — **ADAPT** (largest new machinery)

- **Reused from sglang:**
  - The multi-step draft loop is already captured as a whole in `eagle_draft_cuda_graph_runner.py:86
    EAGLEDraftCudaGraphRunner` (`capture_one_shape` :371, replay :302), replaying the K sequential draft
    steps of `eagle_worker_v2.py:762 draft_forward` (`for i in range(self.speculative_num_steps)`). SSD
    needs the same whole-K-step capture, only with `N = B*MQ_LEN` rows per step instead of topk-expanding
    candidate sets.
  - Per-step KV slicing exists: `eagle_worker_v2.py:714 per_step_draft_out_cache_loc` partitions the decode
    slab per step, and the draft KV slab is already *over-allocated per branch* — `allocation_sizing.py:52-61`
    reserves "worst-case page-aligned footprint per topk branch … reserved for all topk branches" when
    page>1/topk>1 (holey per-branch pages, commit-the-survivor at the end). UNO gets the analogous
    `spec_tokens + 1` clean-root reserve at `:47-51`.
  - The commit side is ready: `spec_utils.py:788 move_accept_tokens_to_target_kvcache` (accepted chain →
    target KV, `seq_lens + num_correct_drafts + 1`).
- **What differs:** the draft attention backend plans each step as shared-prefix decode over slab rows, not
  **per-row variable context lengths + per-row rope offsets + per-row raw prenorm hidden-state chaining**.
  SSD needs a new MQ-style plan mode: per-row `context_lens[depth]`, `rope_positions = base + j_idx + 1`,
  `step_slot_maps` over the per-branch block tables (prototype `_compute_step_positions_and_slot_maps` and
  `_decode_tree_step`), plus hidden-state chaining `/ CC payload["hidden_states"] = prenorm` between steps.
- **New LOC sketch:** attention-backend plan mode (one backend first) ≈ 200–400; per-step slot/rope table
  builder ≈ 60; whole-K-step CG capture shim + payload struct ≈ 120–180. Total ≈ 400–650. (Prototype sizes:
  the plan glue is ~150 LOC across three helpers; the delta is sglang's multi-backend + CG-capture
  invariant cost.)

### 5. Branch cache — outcome-keyed store of finished chains + logits, instant hit-serving — **BUILD-NEW**

- **Reused from sglang:** the *pattern* only. sglang already runs device-resident, req-pool-indexed
  cross-iteration buffers — `overlap_utils.py:255 FutureMap` (publish/resolve lifecycle,
  `new_seq_lens_buf`, `topk_p_buf`) and `overlap_utils.py:177 ConfidenceRelay` (ring buffers + `copy_done`
  CUDA events :211-220) — so a tensor-backed cache with async publication is idiomatic in-repo.
- **What differs:** nothing exists that stores *finished drafter chains + their logits (+ activations)
  keyed by an outcome tuple* `(seq_id, k_idx, rec_token)` and serves them with a vectorized membership test
  in the service path (`hit_cache_and_respond`: `request_keys.unsqueeze(1) == tree_cache_keys.unsqueeze(0)`
  match → slot-fill tokens/logits/activations, no draft forward). The radix cache is token-KV only; there is
  no "return K tokens AND K×V logits without running the drafter" path anywhere.
- **New LOC sketch:** two/three tensor-backed slots + fill + eviction + hit-select ≈ 150–250 (prototype:
  reset 10 lines, populate ≈ 45, hit-serving ≈ 90). Eviction policy (round-robin vs LRU, bounded by
  `MQ_LEN * max_bs` rows) is a small knob.

### 6. "Predicted-outcome" machinery (which draft prefix survives N steps ahead) — **REUSE the relay + survival, BUILD-NEW the outcome prediction**

- **Reused from sglang:** `dspark_planner.py:935 HostConfidenceBudgetPlanner` — `_two_steps_prior_survival`
  (:1017) predicts, *lagged ahead*, how much of the current draft frontier survives its verification round
  (`k_survival = cumprod(lagged_confidence)`, generation-checked :1026-1028; `_shift_to_lag` :998 carries
  lagged slices across iterations), with cost tables from `dspark_sps.py:14 SpsCostTable` /
  `:62 SpsAdditiveCostTable` (`:105 profile_sps_table`, `:152 build_uninitialized_sps_table`), calibration
  via `dspark_sts.py:9 DSparkStsCalibration` + `:39 StsDataRecorder`, and estimator bookkeeping in
  `dspark_block_accept_estimator.py:221 BlockAcceptEstimateRecorder` (online ceiling estimate :100
  `_OnlineCeiling`). The lag relay itself — `overlap_utils.py:130 CONFIDENCE_RELAY_RING_LAG = 2` +
  `ConfidenceRelay` scatter/resolve (:206/:222) — is exactly the two-steps-prior channel SSD's async loop
  needs. `dspark_planner.py:958 compute_budget` shows how to clamp a whole-round budget; applies directly
  to bounding `N = B*MQ_LEN` on memory-poor draft devices.
- **What differs (important nuance):** in SSD the *prediction mechanism is the fork + branch cache itself* —
  the cached chains ARE the predicted outcomes — so DSPARK's survival machinery is a precedent/consumer,
  not a provider. Its useful role for SSD: drive *adaptive fan counts* F per depth (spend more fan on near
  positions when survival is low) around the paper's hard-coded hit/miss fan lists.
- **New LOC sketch:** 0–100 (fan-selector over the budget ≈ 80 LOC if adaptive F is wanted; can lean on
  row 7).

### 7. UNO fixed-budget proposal tree (`uno_tree.py` + `uno_utils.py`) — **ADAPT, optional for MVP**

- **Reused from sglang:** `uno_tree.py:254 _build_tree_kernel` (GPU-native best-first fixed-budget Q
  expansion over pre-fetched per-depth log-prob tables, Triton) + `uno_tree.py:400 build_uno_tree_proposal`
  producing `UnoTreeProposal` (:24) in EAGLE-native `parent_list` / `top_scores_index` ABI that
  `build_tree_kernel_efficient` (eagle_utils.py:156) consumes directly — the header states this intent
  (uno_tree.py:4-9: "expressed directly in EAGLE's candidate-lineage ABI so the existing EAGLE
  implementation can build masks, positions, traversal links, verify …"). `uno_utils.py` provides the
  sampling/plumbing into it (`sample_uno_candidates` :426, `run_uno_sampling` :554, `pack_uno_result` :527,
  sparse-probability helpers :21-:380).
- **What differs:** UNO's tree is a *selection* device over already-computed depth-wise top-rank tables (the
  drafter's regular forward feeds it); it does not launch branches. SSD's fan is hand-launched per-depth
  fan lists. So none of it is needed for SSD's MVP; it becomes the right primitive the moment fan counts
  must be sized dynamically to keep total rows and KV reservation under a device budget (best-first
  log-mass selection instead of hand tables).
- **New LOC sketch:** 0 for MVP; ~100–200 for wiring it as the dynamic fan-selector of row 6.

### 8. KV accounting for a whole SSD round — **REUSE/ADAPT**

- **Reused from sglang:** `allocation_sizing.py:64 get_alloc_reserve_per_decode()` = 2× the per-decode slab
  (:70 "double-buffer that absorbs the kv_committed_len lag in overlap mode"), consumed at
  `eagle_utils.py:1066` in `eagle_prepare_for_decode` (:1057) with `page_aligned_decode_alloc_lens`
  (`allocation_sizing.py:73-93`); allocator sharing across draft/target via `eagle_worker_v2.py:223
  alloc_memory_pool`; req_to_token-row headroom via `allocation_sizing.py:96
  get_req_to_token_extra_context_len` (sized to hold the decode over-allocation + page slack, guarded
  against OOB at `eagle_utils.py:1084-1091`); UNO at `allocation_sizing.py:47-51` shows the per-algorithm
  branch-discriminator precedent. Commit of the survived chain: `spec_utils.py:788
  move_accept_tokens_to_target_kvcache`.
- **What differs:** only the *lookahead count*: SSD's round is `K+1 + K*MQ_LEN` per seq
  (`async_spec_helpers.compute_megaspec_lookahead`), not `spec_steps*topk` / `spec_tokens`. sglang's
  `get_alloc_len_per_decode` (allocation_sizing.py:40-61) reads bag leaves (`max_speculative_num_draft_tokens()`
  + per-algorithm branches), so SSD plugs in as a new branch with no algorithmic KV change.
- **New LOC sketch:** ~20–40 wiring (bag leaf + `SpeculativeAlgorithm.is_ssd` branch at
  `spec_registry.py:84-87` next to `is_uno` / `is_dspark`).

### 9. Plan-stream / overlap plumbing — **REUSE (in-process MVP); BUILD-NEW only for disaggregated draft**

- **Reused from sglang:** `spec_utils.py:1213 get_plan_stream` (env `SGLANG_ENABLE_OVERLAP_PLAN_STREAM`)
  runs verify prep on a side stream — the same primitive lets the branch decode overlap a running target
  verify — with stream-ordering precedent in `eagle_worker_common.py:565 run_eagle_verify` (:602
  plan-stream context; :606 plan waits on the draft frontier; :619 fwd waits on the plan).
  `overlap_utils.py:255 FutureMap` handles publish/relay of cross-iteration values.
- **What differs (out of MVP scope):** the reference engine runs the drafter as a *separate process* driven
  by NCCL pack/recv (`draft_runner.draft_loop`, `draft_async_prefill`, `recv_int64`, fused int64 bursts).
  For an in-process MVP this is unnecessary; if a disaggregated draft service is ever wanted, copy the
  peering precedents (`eagle_disaggregation.py`, `pp_spec_relay.py`) — ~500+ LOC of process glue.
- **New LOC sketch:** 0 for MVP.

## Bottom line

Roughly **2/3 reuse by surface area, ~800–1,300 net-new LOC** for an in-process MVP:

- **Straight REUSE:** KV double-buffer + page-aligned per-branch reserves + accept-to-target commit (row 8),
  the mask ABI + target-side spec verify for the served chain (`eagle_prepare_for_verify`
  eagle_utils.py:522, `run_eagle_verify`; SSD's prepared chain surfaces to the target as an ordinary topk=1
  `build_chain_tree` verify input — eagle_worker_common.py:454), plan-stream (row 9), FutureMap relay
  (row 2), DSPARK relay/survival (row 6).
- **Small ADAPT:** glue decode (~120–200), fork (~80–120), KV lookahead leaf (~20–40), optional adaptive fan
  via UNO (~100–200), target-verify input assembly (reuse `build_eagle_verify_input` with the served chain
  as a depth-(K+1) chain).
- **The genuinely new machinery is concentrated in two places:**
  1. an **MQ-style branch-decode attention plan mode** on the draft side (per-row context lens, per-row rope
     offsets, per-row block-table slot maps + hidden-state chaining + whole-K-step CG capture) — ~300–600
     LOC including the CG capture shim and one attention backend; and
  2. an **outcome-keyed branch cache** with instant hit-serving (~150–250 LOC) — no existing analog, but the
     in-repo `FutureMap`/`ConfidenceRelay` tensor-buffer patterns keep it small.

## Confidence and open questions

**Confidence:** High on rows 1, 2, 3, 7, 8 (verified line-by-line against the current fork and the reference
engine). Medium-high on row 4's estimate: the multi-step CG capture, per-step slab slicing, holey
per-branch page reserves, and commit path are all confirmed, but I did not read each attention backend's
plan function to price the new MQ plan mode per backend — the 200–400 band extrapolates the prototype's
~150 LOC of plan/step glue plus sglang's backend dispatch duplication. Medium on row 5's LOC (simple in the
prototype because it is whole-tensor; multi-round persistence, eviction under `max_running_requests`, and
multi-wave keying are unmeasured).

**Open questions** (candidates for follow-up tickets, none blocking the map):
1. Bonus/termination semantics: confirm how bonus acceptance composes with
   `move_accept_tokens_to_target_kvcache` (`seq_lens + num_correct_drafts + 1`) when the served chain is
   shorter than K.
2. Anatomy of the extend inputs: where do `extend_counts` and the recovery-token activations (reference's
   `target_recovery_activations` / `extend_eagle_acts`) come from on the sglang side?
   `CaptureHiddenMode.FULL` at `eagle_utils.py:599-603` already captures pre-norm target states for verify,
   but FutureMap's `RelayPayload` does not yet relay those activations across the seam — likely a small
   `RelayPayload` field addition.
3. Which attention backend gets the MQ branch-decode plan mode first (FA-varlen on CUDA is the natural
   candidate given the prototype's varlen-context decode interface).
4. Adaptive F vs hard-coded hit/miss fan tables (row 6/7 trade-off; the paper hard-codes them).

## Appendix: symbol → line index

- eagle_utils.py — TreeMaskMode :144 · build_tree_kernel_efficient :156 · eagle_prepare_for_verify :522 ·
  eagle_sample :738 · eagle_prepare_for_decode :1057
- spec_utils.py — select_top_k_tokens :422 · traverse_tree :520 · move_accept_tokens_to_target_kvcache :788 ·
  get_plan_stream :1213
- eagle_worker_common.py — build_eagle_verify_input :395 · run_eagle_verify :565
- eagle_worker_v2.py — alloc_memory_pool :223 · draft_forward :696 (multi-step loop :762) · draft_extend
  :907 · _draft_extend_for_decode :1054 · EAGLEWorkerV2 :1239
- eagle_draft_cuda_graph_runner.py — class :86, capture_one_shape :371 ·
  eagle_draft_extend_cuda_graph_runner.py — class :95
- uno_tree.py — UnoTreeProposal :24 · _candidate_lse_partials_kernel :52 · _build_tree_kernel :254 ·
  build_uno_tree_proposal :400 · uno_utils.py — run_uno_sampling :554 · pack_uno_result :527
- dspark_planner.py — HostConfidenceBudgetPlanner :935 · compute_budget :958 ·
  _two_steps_prior_survival :1017
- overlap_utils.py — CONFIDENCE_RELAY_RING_LAG :130 · RelayPayload :140 · ConfidenceRelay :177 ·
  FutureMap :255
- allocation_sizing.py — get_alloc_len_per_decode :40 (UNO branch :47) · get_alloc_reserve_per_decode :64 ·
  page_aligned_decode_alloc_lens :73 · get_req_to_token_extra_context_len :96
- ragged_verify.py — RaggedVerifyLayout :46 · compute_uniform_extend_lengths :287 ·
  compute_ragged_extend_lengths :307
- dspark_block_accept_estimator.py — _OnlineCeiling :100 · BlockAcceptEstimateRecorder :221
- Reference engine (ssd @ d7eb8fa0): ssd/engine/draft_runner.py — jit_speculate, hit_cache_and_respond,
  _service_spec_request, _build_tree_batch, _decode_tree, _populate_tree_cache, draft_loop;
  ssd/engine/helpers/mask_helpers.py — get_custom_mask(_cached|_vectorized), flat_blocks_after_cat;
  ssd/utils/async_helpers/async_spec_helpers.py — make_glue_decode_input_ids,
  get_forked_recovery_tokens_from_logits, compute_megaspec_lookahead; ssd/engine/block_manager.py —
  may_append(lookahead), ref-counted fork blocks, draft_block_table on is_draft.
