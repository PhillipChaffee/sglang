# SGLang Domain

Glossary for SGLang's serving runtime and the Speculative Speculative Decoding effort tracked on this fork.

## Speculative decoding (from the SSD paper)

**SSD**:
Speculative Speculative Decoding — the scheme this effort adds to SGLang, in which the draft model runs asynchronously on separate hardware, pre-speculating tokens for predicted verification outcomes and serving speculations instantly on cache hits. Paper: arXiv 2603.03251. Reference implementation: tanishqkumar/ssd.
_Avoid_: speculative-2, double speculation

**Target model**:
The large model whose decode quality is authoritative.
_Avoid_: verifier (reserve for the process that runs verification when drafter and verifier are split)

**Draft model**:
The small model that proposes tokens for the target to verify.
_Avoid_: speculative model

**Verification outcome**:
What verifying one speculation round produced: the count of accepted draft tokens plus the bonus token. Written `(k, t*)`.
_Avoid_: verification result, acceptance state

**Bonus token**:
The token the target contributes beyond the accepted draft tokens, drawn from the residual when partially accepted, or from the target distribution when everything is accepted.

**Hit**:
A round whose verification outcome matches one of the drafter's pre-speculated branches, so the next speculation is served from cache with no drafting latency.

**Miss**:
A round whose outcome no branch predicted; the drafter falls back (random tokens or just-in-time speculation).

**Random fallback**:
Serving filler tokens that the target almost certainly rejects, wasting one autoregressive step of the target.

**JIT speculation**:
Fallback in which the drafter speculates just-in-time (K sequential draft steps) instead of serving a pre-built chain.

**Pre-speculation cache**:
The drafter's per-sequence store of pre-computed token chains and logits, one chain per predicted verification outcome, keyed for instant lookup on the outcome report.

**Hypothesis**:
One pre-speculated branch — one predicted verification outcome with its token chain.
_Avoid_: branch (reserve "branch" for the tree-construction act)

**Fan-out**:
The number of hypotheses per lookahead position.
_Avoid_: F, breadth

**Lookahead**:
The number K of draft tokens per speculation round.
_Avoid_: num_steps in SSD context

**Glue decode**:
The drafter's extend pass that absorbs the verified tokens and prior speculation into its KV cache before forking new hypotheses.

**Saguaro sampling**:
Cache-aware draft sampling that biases probability mass toward the already-cached tokens to raise the hit rate.
_Avoid_: sampler_x (implementation knob name)

**Capped geometric fan-out**:
The theoretically optimal allocation of fan-out across lookahead positions, decreasing geometrically with lookahead depth.

**Drafter / verifier split**:
Deployment in which the drafter runs as a separate process (on its own device) from the target-verifying engine, exchanging outcome reports and speculations per round.

## Serving topologies and modes

**Tensor parallelism (TP)**:
Sharding one model's copies of layers across the GPUs of a node (how a 70B model uses 4 GPUs).

**Pipeline parallelism (PP)**:
Splitting a model by layer groups across sequential stages; a speculative worker on PP relays draft trees between stages.

**PD disaggregation**:
A deployment that runs prefill and decoding on separate engines/devices and ships KV state between them.
_Avoid_: PD (ambiguous with drafter/verifier split)

**Drafter/verifier split**:
See split deployment. Distinct concern from PD disaggregation: this splits draft phases from verify phases, not prefill from decode.

**Deployment mode**:
Which of SSD's two drafter deployments an engine runs: in-process (default) or split (opt-in). One config switch selects it, and split refuses the launch cleanly without the drafter device.
_Avoid_: topology (reserved for TP/PP/DP/EP sharding)

**In-process deployment**:
Deployment mode in which the drafter runs inside the target engine's process on the target's own devices; rounds stay synchronous and the pre-speculation cache supplies the asynchrony.
_Avoid_: colocated drafter, local drafter

**Data parallelism (DP)**:
Replicating the model and sharding the incoming requests across replicas; each replica serves whole requests.
_Avoid_: splitting the model (that is model parallelism)

**MLA**:
Multi-head Latent Attention — an attention architecture (DeepSeek, Kimi, MiniMax class) that compresses KV into one shared latent head, keeping per-token KV tiny; it parallelizes badly under TP because that single KV head cannot be sliced.
_Avoid_: MHA/GQA (different attention architectures)

**Expert parallelism (EP)**:
Sharding the MoE experts of a model across GPUs, exchanging tokens between ranks via all-to-all dispatch/combine each layer.

**Data-parallel attention (DP attention / DPA)**:
Making the attention component data-parallel for MLA-class models: each rank owns whole requests and their entire KV cache (never duplicated) plus a replicated copy of the (small) attention weights, while the heavy MoE layers run expert-parallel with all-to-all exchange.
_Avoid_: sharded attention, duplicated KV cache

**Constrained decoding / grammar**:
Generation whose sampling the runtime constrains to a user-supplied grammar (JSON schema or regex).

## SGLang speculative framework

**Spec V2**:
The current speculative-decoding framework: one registered algorithm implements a spec worker that issues draft, target-verify, and draft-extend phases inside `forward_batch_generation`.

**Spec algorithm registration**:
The plugin surface (`SpeculativeAlgorithm.register` + `CustomSpecAlgo`) by which a new algorithm becomes `--speculative-algorithm <name>`.

**Spec worker**:
The scheduler-facing object (subclass of `BaseSpecWorker`) that owns the draft phases and calls target verification.
_Avoid_: draft worker (that is the draft side only)

**Draft worker**:
The object owning the draft model and speculation phases only.
_Avoid_: drafter process (reserve for the split deployment)

**Target verify**:
The phase running the target model over drafted token trees and computing acceptance.
_Avoid_: verification (ambiguous with verification outcome)

**Draft extend**:
The phase re-running the draft model on accepted tokens to seed the next round's hidden states.

**Overlap scheduler**:
The scheduler mode that pipelines GPU work with CPU work; a spec worker must cooperate via `on_publish`, `FutureMap`, and the WAR barrier.

**FutureMap**:
The overlap-scheduler store relaying per-request payloads (accepted tokens, spec extras) from step N to step N+1's scheduling.

**Decoupled spec protocol**:
The (protocol-only today) IPC contract (`decoupled_spec_io.py`) for split drafter/verifier deployments: `DraftSync`, `VerifyCommit`, `DraftTailStreamOutput`, `DraftControlBatch`.

**Spec parity**:
Lossless-output equivalence between a speculative server and the same model served autoregressively; the standard correctness gate for any spec algorithm.
_Avoid_: KL test (no KL-divergence harness exists here)

**Validation pair**:
A target/draft model pair used to validate (e.g. Qwen3-32B + Qwen3-0.6B, Llama-3.1-70B + Llama-3.2-1B).
