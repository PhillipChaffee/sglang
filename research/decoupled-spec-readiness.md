# Research: decoupled-spec protocol readiness for an async SSD drafter

Ticket #2 (map #1) · branch `research/decoupled-spec-readiness` · date 2026-10-12

## Question

Can sglang's decoupled-spec machinery carry an async SSD drafter today, and what exactly is missing?

**Answer in one paragraph:** No — and not close. What landed in upstream commit `f444b5897b` is a schema-only IPC layer: protocol dataclasses with a unit test, plus CLI/PortArgs plumbing for an intended ZMQ mesh. Nothing in the runtime reads the config (`PortArgs.decoupled_spec_ipc_config` has zero downstream consumers outside `server_args.py`), the transport thread the dataclass comments describe (`TokenSyncThread`) does not exist anywhere in the repo, and there is no drafter-side scheduler at all. The existing message types are designed around a different feature (token-ID resync + one-token-at-a-time tail streaming), so SSD's per-round speculation exchange — `O(B)` outcome-key request, `O(B·K·V)` chains+logits response — is simply unrepresentable in the current message set. The good news: the pieces a drafter would *reuse* all exist (draft model runner on its own GPU, draft CUDA graphs, KV pool machinery, a plugin spec-algo registry with a cross-engine draft-input hook, and three in-process/cross-engine precedent patterns for relaying draft results). The build is a new subsystem of roughly 4–6k lines across nine pieces, not an extension of the landed protocol.

Triage note: this run was a single-agent inline investigation (Tier-2 scope; no subagent dispatch available in this harness).

---

## 1. What the landed "decoupled spec" machinery actually is

Upstream commit `f444b5897b` "[Spec][1/N] Decoupled speculative decoding: IPC protocol + cross-process request id + server flags (#27634)" changed exactly 4 files, 804 insertions:

- `python/sglang/srt/speculative/decoupled_spec_io.py` (384 lines, new)
- `python/sglang/srt/server_args.py` (+55)
- `test/registered/unit/spec/test_decoupled_spec_io.py` (272 lines, new)
- `test/registered/unit/server_args/test_server_args.py` (+94)

The "1/N" label is accurate: it is the schema the later N steps were meant to consume.

### 1.1 The message types (verified inventory)

All in `python/sglang/srt/speculative/decoupled_spec_io.py`:

| Type | Lines | Role | Direction |
|---|---|---|---|
| `DraftReqKey` / `build_draft_scheduler_rid` / `parse_draft_scheduler_rid` | 13–40 | cross-process request identity `<rid> = draft:{src_verifier_rank}:{request_id}` | meta |
| `DraftSync` | 43–63 | verify→draft: open/re-open a request; carries `prompt_token_ids` + `committed_outputs` **token id lists** | t→d |
| `VerifyCommit` | 66–105 | verify→draft: commit contiguous output segment (`pre_verify_committed_len`, `committed_tokens`) | t→d |
| `DraftClose` | 108–120 | verify→draft: close a request | t→d |
| `DraftTailStreamOutput` / `DraftTailStreamOutputBatch` | 123–147 | draft→verify: **one output token** per message, tagged with `base_committed_len`, `new_token_pos` | d→t |
| `DraftControlBatch` | 150–155 | coalesced control frame for one drafter rank (sync + commit + close) | t→d |
| `VerifierCommitSegment` | 158–246 | drafter-side coalescing of contiguous commits; contiguity enforced | d-side |
| `DraftControlInbox` / `ReadyDraftControls` | 249–354 | drafter-side inbox with drain + partial-consume semantics | d-side |
| `DraftMeshMessage` / `DraftMeshMessageType` | 357–377 | discriminated envelope: `CONTROL_BATCH` \| `TAIL_STREAM_OUTPUT_BATCH` | both |
| `DecoupledSpecIpcConfig` | 380–384 | `bind_endpoint`, `connect_endpoints`, `rank` (ZMQ endpoints) | config |

Key style observations for the mapping:

- **Everything is token-ID- and Python-object-based.** `DraftSync` resyncs by sending literal token ids (`decoupled_spec_io.py:55-56`); `VerifyCommit` sends literal committed tokens (:84). There is no hidden-state, logits, temperature, block-table, or tensor field on any type.
- **`DraftTailStreamOutput` is a token-streaming feature, not a speculation message.** It carries exactly one token (`decoupled_spec_io.py:142`), exists so the drafter can emit ahead-of-verify tail tokens ("pending-prefix confirmation", :133), and has no logits/hidden fields.
- **Known-unbuilt parts are admitted in-line**: the `VerifierCommitSegment.append_message` docstring names the "transport thread (TokenSync thread at drafter side)" — `rg -ln "TokenSyncThread"` hits only this file and its test — and a TODO at `decoupled_spec_io.py:185-188` says drafter failure should "degrade the verifier into normal autoregressive decoding" — not implemented.

### 1.2 The wiring (verified)

- Flags live in `python/sglang/srt/arg_groups/fields/disagg.py:26-50`: `decoupled_spec_bind_endpoint`, `decoupled_spec_connect_endpoints`, `decoupled_spec_rank`, `decoupled_spec_role ∈ {null, verifier, drafter}`. Doc comment is honest: "currently connected by a ZMQ IPC mesh" (disagg.py:26-27).
- `python/sglang/srt/server_args.py:726` declares `PortArgs.decoupled_spec_ipc_config: DecoupledSpecIpcConfig | None`; built at server_args.py:758-774 (ValueError if role ≠ null and any endpoint/rank missing); threaded into both the plain-IPC `PortArgs` branch (786) and the DP-attention TCP branch (874).
- **Dead-end confirmed**: `rg -n "decoupled_spec_ipc_config"` over `python/sglang/srt` matches only `server_args.py`. No scheduler, worker, model runner, or launch-path code consumes it. Running with `--decoupled-spec-role verifier/drafter` today boots a normal engine that holds unused endpoint strings.
- The test `test/registered/unit/spec/test_decoupled_spec_io.py:1-272` exercises only pure dataflow (id codec round-trip, commit validation, segment coalescing/contiguity, inbox routing, envelope discriminant); its own docstring: "schema-only IPC layer … no GPU or transport here" (test lines 3-8).

---

## 2. The SSD exchange, as actually implemented in the reference engine

Sources: arXiv 2603.03251 ("Speculative Speculative Decoding", Tanishq Kumar, Tri Dao, Avner May; ICLR 2026; the optimized algorithm is named **Saguaro**) — <https://arxiv.org/abs/2603.03251> — and reference engine <https://github.com/tanishqkumar/ssd> (files fetched to scratch for line-level analysis; paths below are repo-relative).

### 2.1 Topology

The driver (`ssd/engine/llm_engine.py`) spawns the drafter as a separate process — literally `ctx.Process(target=DraftRunner, ...)` with spawn context (llm_engine.py:61-86) — target on N GPUs, draft on 1 GPU (README: "70B target (4 GPUs) + 1B draft (1 GPU), k=7 --f=3"). The two processes talk over a dedicated NCCL `dist.ProcessGroup` (`async_pg`); `draft_async` asserts `num_gpus > 1` (llm_engine.py:37-38).

The drafter is a **command-loop process**, not a scheduler: `DraftRunner.draft_loop()` (draft_runner.py:859-933) receives `cmd ∈ {1=prefill, 0=spec_request, 2=exit}` and, on a spec request, does in one synchronous service round:

1. `_service_spec_request()` — recv fused int64 burst, serve from tensor cache;
2. `_build_tree_batch()` — on miss, build the fan-out fork-tree inputs;
3. `_decode_tree()` — run the branch decode;
4. `_populate_tree_cache()` — cache the tree so future rounds hit.

### 2.2 The wire format (extracted from `ssd/engine/speculator_async.py`)

Target→draft, all packed as **fused int64 tensor bursts** (`_speculation_request`, speculator_async.py:143-232):

| payload | shape / dtype | content |
|---|---|---|
| `cmd` | `[1]` int64 (separate send) | 1=prefill / 0=spec / 2=exit |
| `meta` | `[3]` int64 | `(B, K, F=async_fan_out)` |
| `cache_keys` | `[B, 3]` int64 | **outcome key** `(seq_id, last_spec_step_accepted_len − 1, recovery_token_id)` |
| `num_tokens` | `[B]` int64 | confirmed sequence length per request |
| `temps` | `[B]` float32 bit-cast into int64 burst | per-request draft temperature |
| `block_tables` | `[B, max_blocks]` int32 (−1 padded, bit-cast) | **draft-model** block table, managed by the target-side scheduler and pushed every round |
| (EAGLE only) | `recovery_activations` `[B, hidden]` + extend acts/counts/ids | target hidden state for draft conditioning (`fc`) |

Also `prefill()` (speculator_async.py:59-100) sends full `input_ids`, `num_tokens`, `draft_block_table` (+ EAGLE acts) — the drafter's first KV is computed from the target's payload.

Draft→target (speculator_async.py:228-231 and `hit_cache_and_respond`, draft_runner.py:186-286):

| payload | shape | content |
|---|---|---|
| `fused_response` | `[B + B·K]` int64 | **`cache_hits` bitmap `[B]`** then **speculation chains `[B, K]`** |
| `logits_q` | `[B, K, V]` (draft dtype) | draft logits for the K chains — the verifier doses rejection-sampling correction from these (`ssd/utils/verify.py`) |
| (EAGLE only) | `out_activations` `[B, K, hidden]` | |

That is the ticket's `O(B)` request and `O(B·K·V)` response, verbatim. Typical bulk: B=8, K=7, V≈151k, fp16 ⇒ ~16–34 MB per round-trip — exactly why the reference engine uses NCCL device-tensor sends and why a Python-object ZMQ channel cannot be the vehicle for this message.

### 2.3 The speculation cache and forks

- Cache is a **tensor-backed key/value store on the drafter**: `tree_cache_keys [T,3]`, `tree_cache_tokens [T,K]`, `tree_cache_logits [T,K+1,V]`, `tree_cache_activations [T,K,hidden]`; membership is a vectorized `eq`/`match.any(dim=1)` against the incoming `[B,3]` keys (draft_runner.py:218-241). The `k−1` component (`last_spec_step_accepted_len − 1`, seeded −1 to force a first-round miss, sequence.py:36; scheduler_engine.py:265 sets it after each verify) plus `recovery_token` identify the *predicted verify outcome* the tree was built for.
- Between service rounds the drafter **pre-builds projection trees**: a "glue decode" of `[B, K+1]` tokens (recovery token first, then K chains — `make_glue_decode_input_ids`, async_spec_helpers.py), then forks `F` branches per position from top-`F` logits (`get_forked_recovery_tokens_from_logits`; per-position fan-out widths `fan_out_t` / `fan_out_t_miss`; tree span `compute_megaspec_lookahead = K + 1 + K·MQ_LEN`). These fork KV rows live in the **target-managed** drafter block table (target-side scheduler pre-reserves lookahead capacity: `BlockManager.may_append(seq, lookahead)` block_manager.py:145-164; post-verify `_update_kv_caches` walks back to the accepted index — scheduler_engine.py:285-315). Forked blocks share prefix pages with ref counts (block_manager.py:83).
- Sampler quirk relevant to correctness: `apply_sampler_x_rescaling` (async_spec_helpers.py:96-127) rescales top-F draft probs — SSD's corrected sampling differs from vanilla rejection sampling, so the verifier's sampling math must be ported with the algorithm, not assumed from sglang's EAGLE verify path.

### 2.4 Who owns state (important divergence decision)

In the reference engine the **target-side driver owns the drafter's block manager** and pushes `[B, max_blocks]` tables each round, keeping the drafter nearly stateless. A sglang drafter would instead run its own scheduler with its own `ReqToTokenPool` (sglang's natural architecture, visible in the control-inbox design), so the control channel could carry only `O(B)` lengths/temps — **dropping the block tables entirely**. That is a deliberate protocol divergence from the reference worth a decision ticket before build (open question §9.2).

---

## 3. Mapping SSD's exchange onto the landed protocol types

| SSD wire item | Dir | Landed sglang type | Verdict |
|---|---|---|---|
| `cmd` frame (1/0/2) | t→d | `DraftMeshMessage` envelope has only two discriminants (`CONTROL_BATCH`, `TAIL_STREAM_OUTPUT_BATCH`), no command/op frame | **extend** |
| full-token prefill payload (+ block table, + acts) | t→d | `DraftSync` carries prompt+committed **token ids only** | **partial**: alignment intent matches (docstring "align to before it can emit …"), but no EAGLE-act/hidden-state field, no temperature |
| outcome key `(seq_id, k−1, recovery)` `[B,3]` | t→d | nonexistent. `DraftReqKey` is request *identity*, not *outcome*. The data to derive keys exists (VerifyCommit lengths + tokens), but `VerifyCommit` is a resync control, not a speculation rally keyed into a drafter cache | **MISSING**: new `SpeculationRequest` message |
| `num_tokens`, `temps` `[B]` | t→d | lengths derivable from `VerifyCommit`; **temperature**: absent from every landed type | **MISSING** fields |
| `block_tables` `[B, max_blocks]` | t→d | nothing; likely *intentionally omitted* in favor of drafter-self-managed KV (§2.4) | divergent-design decision |
| `recovery_activations` `[B, hidden]` | t→d | nothing; no tensor/hidden field exists | **MISSING** |
| `cache_hits` `[B]` + chains `[B,K]` | d→t | closest is `DraftTailStreamOutput` — but **one token per message** (:142), no batch, no bitmap, tail-streaming semantics | **MISSING**: new `SpeculationResponse` type |
| `logits_q` `[B,K,V]` (+ acts `[B,K,hidden]`) | d→t | nothing; **transport mismatch**: `DecoupledSpecIpcConfig` is a ZMQ-endpoints config; a Python-object ZMQ mesh cannot carry multi-MB GPU tensors without D→H copies | **MISSING** message + bulk tensor channel |
| verify result → next-round commit | t→d | `VerifyCommit` **exists and fits**: contiguous committed segment ≡ SSD's `VerifyResult.new_suffixes` | **EXISTS** |

So of the landed cast: `DraftSync` ≈ prefill/open (extend for acts/temps), `VerifyCommit` ≈ commit path (usable as-is), `DraftClose` fine, `DraftControlBatch`/`DraftMeshMessage` = usable-but-unproven envelopes needing a second message family, `DraftTailStreamOutput` = unrelated feature (orthogonal), `DecoupledSpecIpcConfig` = unproven transport config with zero consumers.

---

## 4. Where the existing runtime could carry an async drafter (verified precedents)

Three in-repo patterns cover variants of "a draft result arrives from somewhere else"; none is wired cross-process today, but each maps one piece of an SSD integration:

1. **FutureMap / RelayPayload** (`python/sglang/srt/managers/overlap_utils.py`) — pool-indexed, same-GPU tensor relay: the forward pass *publishes* per-request spec extras (`RelayPayload.from_draft_input` publishes bonus tokens, topk_p, topk_index, hidden states, draft_probs — overlap_utils.py:139-174), and the next iteration resolves them (`resolve_forward_inputs`, overlap_utils.py:87-127; `FutureMap.stash` :603). This is how today's verifier "consumes" the last iteration's draft outputs — in-place, by `req_pool_idx`, zero-copy, generation-guarded. A cross-process analog would be an IPC-fed FutureMap: same resolve seam in the scheduler, buffers filled by the transport thread instead of the forward pass.
2. **Cross-engine draft state at engine handoff** (`python/sglang/srt/speculative/eagle_disaggregation.py`, plus `dflash_disaggregation.py`, `dspark_disaggregation.py`) — proven pattern for building a `SpecInput` on the consumer side from per-request relayed state (`build_eagle_disagg_draft_input` rebuilds an `EagleDraftInput` from req-borne `output_topk_p`/`output_topk_index`). Works only for request-object metadata at admission (prefill→decode), not per-round bulk tensors — but it *is* the template for "draft state born elsewhere becomes a verify input".
3. **Algorithm-agnostic tree relay across stages** (`python/sglang/srt/speculative/pp_spec_relay.py:10-78`) — `PPSpecRelayInput(SpecInput)` carries a draft tree (tokens+parents) pinned on `ScheduleBatch.spec_info` with filter/merge hooks, riding the PP ring. Its docstring: "A speculative algorithm whose proposal is not an EAGLE-style tree can subclass it (or mirror it) and only has to supply its own rebuild" (pp_spec_relay.py:22-26). An SSD chain bundle (recovery + K tokens + topology) fits this subclass shape; `scheduler_pp_mixin.py:1091-1297` shows relay ingress/egress behind the same batch seam.

And the plugin seam: `python/sglang/srt/speculative/spec_registry.py` lets SSD register as a custom spec algorithm (`register_algorithm`, spec_registry.py:234-264) with a `build_disagg_draft_input` hook (spec_registry.py:164-175) meant exactly for building draft inputs from cross-boundary state, plus duck-typed `has_draft_kv` (114-116). No code today combines `decoupled_spec_role` with a speculative algorithm — that bonding doesn't exist yet.

## 5. Capability inventory

### EXISTS (proven)

- Protocol schema + cross-process rid codec + flags + PortArgs threading + CPU unit tests (`decoupled_spec_io.py`; `server_args.py:758-774, 786, 874`; commit `f444b5897b`).
- ZMQ socket plumbing *patterns* in-engine (scheduler↔tokenizer, PD bootstrap, metrics channels) — a transport thread could borrow the style; the machinery itself is absent.
- Draft model runner constructible on its own GPU: `EagleDraftWorkerBase` builds separate runners with their own memory pools, attention backends, CUDA graphs (`base_spec_worker.py:56-114`: `alloc_memory_pool` / `init_attention_backends` / `init_cuda_graphs`), and `BaseSpecWorker._draft_model_runners` (:168).
- Draft CUDA graphs for chain decode: `eagle_draft_cuda_graph_runner.py` (752 lines) + `_rebuild_topk1_chain_buffers` (base_spec_worker.py:115).
- Verify machinery: `EAGLEWorkerV2.verify` (eagle_worker_v2.py:1907), `ragged_verify.py`, verify-as-fixed-length-extend; `draft()` (eagle_worker_v2.py:614), `draft_extend` (:907).
- Per-request draft KV pool, radix-cached, fork-capable prefix sharing (ReqToTokenPool + paged allocators + RadixCache under `sglang/srt/mem_cache/`).
- Plugin spec-algo registry with cross-engine hook (`spec_registry.py` above).
- Scheduler event-loop seams: `event_loop_normal` (scheduler.py:1940-1972) / `event_loop_overlap` (:1975-2048) — `ingest_requests → get_next_batch_to_run → run_batch → process_batch_result`; process entry `run_scheduler_process` (scheduler.py:6060). A drafter event loop is an ingest-from-inbox variant of this shape.
- Standalone draft EAGLE (own lm_head, no target-shared embeddings): `standalone_worker_v2.py:24` — in-process, but establishes "draft pipeline independent of target internals".

### EXISTS-BUT-UNPROVEN (machinery present, never exercised end-to-end)

- The ZMQ IPC mesh itself: endpoints configured, **no send/recv code anywhere reads `DecoupledSpecIpcConfig`** — sockets have never been opened.
- `DraftControlInbox` / `VerifierCommitSegment` control flow incl. partial-consume buffering: unit-tested dataflow, never run inside any scheduler event loop (the inbox API assumes a "drafter scheduler [that] extracts and consumes them each time it finishes a decoding step" — decoupled_spec_io.py:254 — which doesn't exist).
- `DraftMeshMessage` envelope + `DraftTailStreamOutput` streaming path: no producer/consumer wired ("verifier-side DraftTailBuffer" named in the docstring, decoupled_spec_io.py:126, has no definition in the repo).
- Failure/degradation semantics: TODO at decoupled_spec_io.py:185-188 (quarantine per-request violations; degrade verifier to AR on drafter failure).
- `PPSpecRelayInput` across stages (end-to-end in-process via the PP ring, but never cross-*process* over IPC).
- FutureMap spec-extras resolve across iterations (`_resolve_spec_extras`, overlap_utils.py:124-127) — same-GPU only; a cross-process version unproven apart from the design precedent.

### MISSING (nothing in the repo attempts it)

1. **Transport**: a `TokenSyncThread`-class transport per role reading `DecoupledSpecIpcConfig` (control channel) **and** a bulk tensor channel for `[B,K,V]` logits/acts (NCCL process group across the two engines, as SSD uses; or a /dev/shm ring). Neither exists; a ZMQ-object mesh is wrong for the bulk leg.
2. **Speculation message family**: `SpeculationRequest` (keys/nums/temps [+acts]) and `SpeculationResponse` (hit bitmap, K chains, logits_q [+acts]). New dataclasses + envelope modes; fused tensor/int64 packing discipline borrowed from SSD's pre-allocated buffers.
3. **Outcome-key speculation cache** (drafter): tensor-backed `keys[T,3]` / `tokens[T,K]` / `logits[T,K+1,V]` (+acts), vectorized membership hit path, eviction policy. ~250–400 lines modeled on draft_runner.py:186-290.
4. **Drafter process/event loop**: `run_draft_scheduler_process` + `DrafterScheduler` — a scheduler.py:1940-2048-shaped loop whose ingest is the control inbox and whose run step is (glue decode | fork-tree decode) → respond; self-owned block manager (§2.4); preemption = force key `k−1 = −1` cache miss (reference behavior).
5. **Drafter fork-tree runner**: EAGLE3-style draft with `fc(target_acts→hidden)` conditioning; glue decode `[B,K+1]`; fan-out `F` per position as ragged extend (per-branch positions/rope/slot maps, packed varlen); fork KV rows sharing prefix pages with refcounts; tree→cache population. The reference did all of this in `draft_runner.py` ≈ 1.1k lines.
6. **Verifier integration**: spec-request dispatch from the verify path + **blocking replay-style wait** (the reference *blocks* on the reply each round — its asynchrony comes from the pre-built cache, not non-blocking scheduling) + adapter `SpeculationResponse →` verify input (mirror `PPSpecRelayInput` shapes) + port `ssd/utils/verify.py` sampling (draft temp + `sampler_x` correction, async_spec_helpers.py:96-127) — noting sglang's verify path currently assumes standard rejection sampling on EAGLE tree tokens.
7. **CUDA-graph path for fork decode** (or declared v0-eager): fan-out widths differ per hit/miss per row ⇒ ragged widths; pad to worst-case `fan_out_t_miss` for static capture with dims `(B, K, F)`; extend `eagle_draft_cuda_graph_runner.py`, else run eager v0 (draft models are small).
8. **Failure/launch/orchestration**: degrade-to-AR (closing the TODO), two-engine launch + GPU assignment + role↔spec-algo validation, warmup.
9. **Sampling-state plumbing**: draft temperature per round has no message field; either extend the request family with drifting temps or rely on a drafter-local Req store mirrored at `DraftSync`.

---

## 6. Build list with size guesses

Order is build order (each layer most useful next):

| # | Piece | Est. size | Notes |
|---|---|---|---|
| 1 | Transport thread + bulk tensor channel | 600–900 LOC | TokenSyncThread per role over ZMQ (control) + NCCL PG or SHM (bulk); mirrors PD-disagg bootstrap socket patterns |
| 2 | Speculation message family + protocol v2 types | 200–350 LOC | new dataclasses, envelope modes, temp/act fields; keep landed control types untouched |
| 3 | Spec plugin registration (`ssd` algo stub) | 150–300 LOC | `register_algorithm` incl. has_draft_kv/build_disagg_draft_input; server-args validation |
| 4 | Outcome-key speculation cache | 250–400 LOC | tensor-backed; unit-tested |
| 5 | Drafter scheduler process + event loop | 800–1,200 LOC | inbox ingest, self-managed draft KV, preemption, respond; reuse scheduler seams |
| 6 | Drafter runner: glue decode + fork-tree decode + conditioning + tree population | 1,000–1,500 LOC | port/adapt ssd draft_runner.py; ragged fork extend on sglang backends |
| 7 | CUDA graphs for fork decode (or v0-eager) | 400–700 LOC | padded fan-out capture dims (B, K, F); extend eagle_draft_cuda_graph_runner |
| 8 | Verifier integration + verify-math port + fallback | 600–900 LOC | dispatch/recv, SpeculationResponse→verify-input adapter, sampler_x/draft-temp port, AR fallback (closes the decoupled_spec_io.py TODO) |
| 9 | Launch/orchestration + role validation | 200–400 LOC | two-engine launch, GPU pins, warmup |

**Total: ~4.2k–5.9k LOC** — ~40% (transport, scheduler, launch) borrows sglang infrastructure; ~30% (fork-tree runner, verify math) ports ~2k proven reference-engine lines; ~30% is genuinely new glue (message family, cache, adapters). Sizing calibrated on the reference engine's analogous files (draft_runner.py ≈ 45KB, speculator_async.py ≈ 9KB, scheduler.py ≈ 16KB) and sglang's comparable modules (eagle_worker_v2.py 1,922 LOC, eagle_draft_cuda_graph_runner.py 752 LOC). Sizes are rough: ±30%.

## 7. How the verifier would consume a cache-hit response (direct sub-question answer)

- **Today, in-process** (same engine): the verifier consumes spec extras through `FutureMap`/`RelayPayload` (overlap_utils.py:87-127, 139-174): tensors land in pool-indexed buffers during one iteration and are resolved by `req_pool_idx` next iteration. A cache-hit response would be another stash *source* — but same-process only.
- **Cross-engine today** (PD disagg): `build_eagle_disagg_draft_input` (eagle_disaggregation.py:48) rebuilds spec inputs from request-object state carried at admission — the precedent for "draft state born elsewhere becomes an EagleDraftInput", but once-per-request (prefill→decode), not per-round.
- **Missing for per-round consumption**: a polled IPC inbox in the scheduler event loop (ingest at `get_next_batch_to_run`), an adapter `SpeculationResponse → SpecInput` subclass (mirror `PPSpecRelayInput`), and a blocking-vs-nonblocking-wait decision. The reference engine *blocks* on the replay (`dist.recv` in `_speculation_request`, speculator_async.py:228-231) — asynchrony comes from the pre-built cache making the reply instant, so v0 integration does **not** require a non-blocking overlap-scheduler rewrite; a synchronous `event_loop_normal`-style verifier can integrate first.

## 8. Confidence

- **HIGH**: schema-only status of the landed machinery (zero consumers of `decoupled_spec_ipc_config`; `TokenSyncThread` absent; commit `f444b5897b` stats); presence/absence of every runtime capability listed above; the reference engine's wire format (extracted directly from its source, fetched copies in scratch); verifier-blocking semantics of the reference implementation.
- **MEDIUM**: size estimates (±30%); "verifier could drop block tables entirely with drafter-self-managed KV" (a design divergence derived from the reference's stateless drafter + sglang's scheduler-owned convention — architecturally sound but unvalidated by any prototype).
- **MEDIUM-LOW**: exact build-order (transport-before-scheduler could flip); performance viability of ZMQ for the control leg under load.

Sources used:
- Code: all paths inline above (sglang fork at commit `45418d7dc6`; upstream protocol commit `f444b5897b`).
- Paper: <https://arxiv.org/abs/2603.03251> (v3, ICLR 2026; Saguaro; 30% faster than optimized spec baselines, up to 5× vs AR).
- Reference engine: <https://github.com/tanishqkumar/ssd> — README (async benchmark shape, `k=7 --f=3`), plus `ssd/engine/speculator_async.py`, `ssd/engine/draft_runner.py`, `ssd/engine/scheduler.py`, `ssd/engine/block_manager.py`, `ssd/engine/sequence.py`, `ssd/engine/verifier.py`, `ssd/utils/verify.py`, `ssd/engine/helpers/speculate_types.py`, `ssd/utils/async_helpers/async_spec_helpers.py` (all fetched 2026-10-12).

## 9. Open questions (for follow-on ticket #6)

1. **Bulk-channel transport**: NCCL `dist.ProcessGroup` spanning two independently-launched engines (reference approach) vs /dev/shm ring vs ZMQ-with-shm; interacts with sglang's NCCL port allocation (`PortArgs.nccl_port`) and multi-node questions (SSD's roadmap lists per-device draft-DP on up to 4 devices — README "Roadmap").
2. **Drafter-owned vs target-owned draft block tables** (§2.4): sglang's natural architecture puts the block manager drafter-side (protocol drops tables, keeps strict `O(B)` control width) — but diverges from the reference engine and from `DraftSync`'s "verifier is the source of truth" framing (decoupled_spec_io.py:47-49).
3. **Logit transport size**: full-vocab `logits_q [B,K,V]` (~16–34 MB/round fp16/fp32 at V≈150k) — accept NCCL bulk vs trim to acceptance-sufficient statistics; also whether EAGLE-act conditioning is in v0 scope at all.
4. **Blocking vs nonblocking verify wait**: reference blocks on the drafter reply; sglang's overlap scheduler could instead poll a spec-response inbox — is the extra machinery worth it given hit-latency bound? (Suggest: v0 = blocking like the reference; overlap integration later.)
5. **Verify-math integration**: SSD's cached-logit `sampler_x` rescaling vs sglang rejection-sampling verify — port as a special verify-input class, or generalize? (Biggest correctness-sensitive port; `ssd/utils/verify.py` ≈ 7.8KB.)
6. **Protocol membership of tail tokens**: the landed `DraftTailStreamOutput` tail-streaming feature (not needed by SSD v0) — does the upstream "1/N" plan slot it before or after a speculation message family? Re-triage against upstream progress when building.
