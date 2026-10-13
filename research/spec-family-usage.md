# Research: which spec families do users actually run (sequencing evidence)

**Ticket:** PhillipChaffee/sglang#5 (map: #1) · **Research date:** 2026-10-12
**Question:** Rank speculative-decoding families by the strength of evidence that real users run them, so SSD's family-support order follows demand.

Families ranked: EAGLE/EAGLE3 (incl. NEXTN alias / DeepSeek MTP), standalone small draft LLM (STANDALONE), DFLASH (draft layers carved from the target), DSPARK (DeepSeek-V4 self-drafting blocks), UNO, frozen-KV MTP, NGRAM, plus SSD itself as an existing-demand track.

## Caveat on all numbers below

GitHub issue-search counts are proxies, not telemetry: they measure who mentions a family (often because the family *broke*), and release/docs/cookbook billing measures what upstream promotes. Two systematic biases apply and are corrected where noted: (1) full-text issue search counts (title+body+comments) are inflated by roadmap/tracking issues that list many families at once — title-only counts are used for ranking; (2) the EAGLE umbrella and MTP/NEXTN naming overlap, so EAGLE-family sub-counts are reported together. "SSD" as a search term collides with SSD-drive storage (HiCache/Mooncake) upstream — release-note "SSD" hits were manually verified to be all storage, zero algorithm.

## Rank table (strong demand first)

| Rank | Family | Title-matched issues (engagement, created ≥ 2026-07) | Cookbook `--speculative-algorithm` occurrences | Release-note billing (v0.5.12–v0.5.21) | Confidence |
|------|--------|------------------------------------------------------|-----------------------------------------------|----------------------------------------|------------|
| 1 | EAGLE/EAGLE3 umbrella (incl. NEXTN/MTP) | 151 (405; 36 recent) — EAGLE 50, EAGLE3 60, NEXTN 41 | 398 of 543 (~73%) — EAGLE 218, NEXTN 137, EAGLE3 43 | In every release; header billing | **High** |
| 2 | DSPARK | 51 (237; 49 recent) | 76 | Largest recent-series billing; flagship team blog | **High** (rank), **Medium** (magnitude — surge may track DeepSeek-V4 deploy volume) |
| 3 | DFLASH | 29 (90; 18 recent) | 52 | Regular billing incl. dedicated subsections | **Medium-High** |
| 4 | NGRAM | 8 (19; 2 recent) | 8 | Occasional; "no extra model" fallback slot in docs | **Medium** |
| 5 | STANDALONE | 3 (4; 0 recent; 51 loose full-text) | 5 | Occasional | **Medium** |
| 6 | UNO | 0 (9 body-match total; 1 discussion) | 4 | Rare (1 in v0.5.20) | **Medium**, with a flagged contradiction (see below) |
| 7 | Frozen-KV MTP | 0 (1 dedicated bug) | 0 | 1 fix (v0.5.12) | **Low** (small but real) |
| — | SSD (algorithm) | 0 (1 feature request) | 0 | 0 hits | **High** that demand is ~nil |

Sources for the numbers: GitHub issue search `[repo:sgl-project/sglang type:issue in:title <term>]`, e.g. [DSpark](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+dspark&type=issues), [EAGLE3](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+EAGLE3&type=issues), [dflash](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+dflash&type=issues); cookbook flag counts measured over `docs/src/` of `sgl-project/sglang@main` on 2026-10-12 (method: `rg --speculative-algorithm`); release counts over the [release notes](https://github.com/sgl-project/sglang/releases).

## Per-rank evidence

### Rank 1 — EAGLE/EAGLE3 umbrella (incl. NEXTN / MTP). Confidence: High

- **Docs promote it first and most.** The main doc's own quick guidance: "Best speed/quality (recommended): Use EAGLE-3"; "Strong default / broad compatibility: Use EAGLE-2" — [docs/advanced_features/speculative_decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) ([source .mdx in repo](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx)). EAGLE/adaptive-EAGLE are also the only two algorithms the [adaptive speculative decoding](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx) doc supports.
- **Cookbook deployment snippets are dominated by the family**: 218 EAGLE + 137 NEXTN + 43 EAGLE3 = 398/543 (~73%) of all `--speculative-algorithm` values across `docs/src/` (DeepSeek, Qwen, GLM, Kimi, Llama, MiniMax, Nemotron pages).
- **SpecBundle is an all-EAGLE3 industry program** (Ant Group, Meituan, Nex-AGI, EigenAI; lmsys HF collection), whose own page admits "adoption … especially EAGLE3 — remains limited" and exists to fix exactly that — [SpecBundle Usage](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/specbundle/specbundle_usage.mdx), [Supported Models](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/specbundle/supported_models.mdx), [HF SpecBundle collection](https://huggingface.co/collections/lmsys/specbundle).
- **Issue traffic is the broadest**: 151 title-matched issues (EAGLE 50, EAGLE3 60, NEXTN 41) and 19 discussions mentioning "eagle" vs ≤5 for every other family — [discussion search](https://github.com/sgl-project/sglang/discussions).
- **Recency shape**: raw EAGLE3 issues skew old (median creation 2026-01) while NEXTN is rising (18 issues since 2026-07, median 2026-06) — the umbrella's current traffic is mostly vendor-bundled MTP drafting rather than user-fetched EAGLE draft checkpoints.

### Rank 2 — DSPARK. Confidence: High on rank; Medium on magnitude

- **Hottest current activity of any family**: 51 title-matched issues, the highest engagement of any title set (237 comment+reaction sum), and 49 of a 50-issue page created since 2026-07 — users actively report running it on DeepSeek-V4(-Flash/-Pro) (#43486 OOM, #43738 prefix-cache regression), DSV4.1 on 2×DGX Spark (#33289), LFM2 (#39262), Kimi K3 (#40045). [Search](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+dspark&type=issues)
- **Upstream's flagship spec blog is DSpark**, not any other family — "DSpark in SGLang" (2026-07-06) with DeepSeek-V4-Flash benchmarks, a public roadmap tracker (#30344), and reproducible commands — https://www.lmsys.org/blog/2026-07-06-dspark-sglang
- **Release billing is largest**: DSPARK 8 (v0.5.19), 10 (v0.5.17), 11 (v0.5.18), 12 (v0.5.20), 8 (v0.5.21) — [v0.5.21 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)
- **Cookbook presence 76 occurrences** (DeepSeek-V4 pages lead; LFM2-VL page features DSpark with "1.66x to 2.56x speedup" — [PR #40651](https://github.com/sgl-project/sglang/pull/40651)).
- Caveat: much of the issue volume follows the DeepSeek-V4 deployment surge (bugs = people running it in production); third-party posts confirm practitioner interest ([DSpark and DFlash: 10 Notes](https://kylehh.github.io/code25/2026/07/17/dspark-dflash/), [cost-math on when DSpark/EAGLE/DFlash/MTP pay](https://ai-cost-estimator.com/blog/speculative-decoding-cost-math-dspark-eagle-dflash-mtp-when-it-pays)).

### Rank 3 — DFLASH. Confidence: Medium-High

- **29 title-matched issues, 18 since 2026-07** — a distinct user base running it, visible as bug reports: DFlash2 state corruption (#36548), DFlash + Qwen3.8-27B accept-rate → 0 (#43514), no PD-disaggregation support (#36140, 7 comments). [Search](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+dflash&type=issues)
- **Vendor-bundled drafter checkpoints are a strong adoption vector**: Poolside Laguna ships `poolside/Laguna-S-2.1-DFlash-*` ([cookbook config](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/poolside/laguna-s21.jsx)), Xiaomi MiMo-V2.6 ships a bundled `dflash/` drafter ([config](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/XiaomiMiMo/mimo-v2.6.jsx)), and Kimi K3 support landed as "[Spec] Support DFLASH for Kimi K3" ([PR #40794](https://github.com/sgl-project/sglang/pull/40794)).
- **Sustained engineering + extension surface**: LiLiCorr reranking for DFlash drafts ([PR #37462](https://github.com/sgl-project/sglang/pull/37462)), out-of-tree DFlash extension points ([PR #38740](https://github.com/sgl-project/sglang/pull/38740)) — both in [v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) — plus its own dedicated doc section ([DFLASH docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)).
- 52 cookbook `--speculative-algorithm` occurrences.

### Rank 4 — NGRAM. Confidence: Medium

- 8 title-matched issues but **a live recent uptick**: SGLANG_NGRAM_FORCE_GREEDY_VERIFY bug (#43931), ROCm DSA crash (#43730), token-id-0 over-sampling (#43725) — all created 2026-10-10..12; plus #43412 from 2026-10-09. [Search](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+ngram+speculative&type=issues)
- Docs slot it as the zero-dependency choice: "**No extra model available**: Use NGRAM" ([docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)); 8 cookbook occurrences.
- Mixed release fortune: occasional fixes alongside cleanup commits ("drop dead ngram attrs", [v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)) — in active use but not a flagship.

### Rank 5 — STANDALONE (small draft LLM). Confidence: Medium

- Docs carry a full dedicated section ("Standalone Speculative Decoding (Small Draft Model)") and it is the reference algorithm for the spec-v2 overlap-scheduler example ([docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)); 5 cookbook occurrences.
- But actual user traffic is thin: only 3 title-matched issues (median 2026-05), 4 engagement. The 51 full-text hits are mostly places the flag option is listed. [Search](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+in%3Atitle+STANDALONE+speculative&type=issues)

### Rank 6 — UNO. Confidence: Medium, with a flagged contradiction

- Full docs section with two sub-modes ([docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)) and a dedicated benchmark runner ([benchmark/uno README](https://github.com/sgl-project/sglang/blob/main/benchmark/uno/README.md)); DeepWiki's implementation survey lists it prominently ([deepwiki spec page](https://deepwiki.com/sgl-project/sglang/12-speculative-decoding)).
- But **zero title-matched issues** and only 1–2 discussions — the clearest "documented but little user traffic" case among the families. The benchmark runner's own framing ("AR, UNO, DFLASH, EAGLE, or EAGLE3" as peer options) suggests developer evaluation rather than production runs. [Discussion search](https://github.com/sgl-project/sglang/discussions)

### Rank 7 — Frozen-KV MTP. Confidence: Low (small but real)

- Not in any docs feature page; user-reachable only via maintenance traces: fixed once in [v0.5.12](https://github.com/sgl-project/sglang/releases/tag/v0.5.12) ("Fix frozen-KV MTP crash when bonus_tokens is None", [PR #25204](https://github.com/sgl-project/sglang/pull/25204)) and reported by a user in #32666 ("Frozen-KV MTP corrupts output at concurrency ≥ 2… depth 4 also fails verify-graph capture", closed). DeepWiki's survey lists it among "alternative approaches" ([deepwiki](https://deepwiki.com/sgl-project/sglang/12-speculative-decoding)).
- Zero cookbook flags; zero dedicated feature docs.

### SSD (as an existing-demand track). Confidence: High that demand is ~nil

- Exactly one external feature request upstream: [#19896 "[Feature]: Support Speculative Speculative Decoding (SSD)](https://github.com/sgl-project/sglang/issues/19896) (2026-03-04). Timeline: two non-maintainer volunteers showed up ("I am going to try and implement this" — [comment](https://github.com/sgl-project/sglang/issues/19896#issuecomment-4000660359); "happy to contribute" — [comment](https://github.com/sgl-project/sglang/issues/19896#issuecomment-4096132312)); a collaborator said SSD requires "the whole stack, thousands of lines" and a roadmap first ([comment](https://github.com/sgl-project/sglang/issues/19896#issuecomment-4002069231)); the issue was then **auto-closed as inactive on 2026-05-20** ([bot comment](https://github.com/sgl-project/sglang/issues/19896#issuecomment-4493491060)). No upstream PR, no roadmap, no release billing ever followed.
- "SSD"-word searches upstream resolve to SSD-*drive* storage (HiCache/Mooncake offload), not the algorithm — e.g. [v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) "Fix silent SSD offload corruption…". No other algorithm-level SSD demand surfaced in issue search ([SSD speculative search](https://github.com/search?q=repo%3Asgl-project%2Fsglang+type%3Aissue+SSD+speculative&type=issues)).

## Impact on SSD family-support ordering

1. **A DFLASH-style draft ranks third in demand.** It should sit behind the EAGLE umbrella and DSPARK, and ahead of NGRAM/STANDALONE/UNO/frozen-KV MTP: it has the strongest vendor-packaged checkpoint path (Poolside, Xiaomi, Kimi K3), a real-but-smaller issue base (29 issues / 90 engagement), and upstream is actively extending it (LiLiCorr, out-of-tree extension points). If SSD supports three draft families at first, DFLASH is the natural #3 candidate — with the caveat that its draft is a separate checkpoint, like EAGLE.
2. **A standalone-drafted SSD has no measured user demand as-is** (rank-5 family, 3 issues), and SSD's own demand track is ~nil. The *combination* therefore ranks low absent new evidence — but if the standalone family is chosen for SSD early-rollout reasons (self-contained, no vendor coupling), that is a bet upstream is not yet signaling.
3. **Ordering implication**: if SSD's first supported family follows current demand, the sequence is (1) EAGLE/MTP-style drafts — biggest fleet, and NEXTN momentum is that family's freshest slice; (2) DSPARK — loudest current engagement and the family upstream most recently productized; (3) DFLASH. Everything else follows the rank table.

## Open questions

- No public user survey or telemetry exists for SGLang spec-family usage; all evidence is third-party-explained proxies (bugs, blogs, docs, bundling). A usage poll in sgl-project/sglang discussions is the unexecuted instrument that would settle ranks 2–6 directly.
- Is DSPARK's surge technique-driven or a transient effect of the DeepSeek-V4/V4.1 deployment wave? If SSD targets pitchable async overlap generically, the answer changes whether DSPARK is a must-have early integration or a co-equal tier-2.
- Do NEXTN/MTP deployments mostly run `--speculative-algorithm EAGLE` under the hood (the docs MTP section does)? If so, umbrella rank 1 understates EAGLE-engine usage and further de-prioritizes non-EAGLE families relative to NEXTN.
