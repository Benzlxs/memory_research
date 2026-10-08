# Memory-management ideas for continual learning: a low-GPU shortlist

*Brainstorm dated 2026-10-08. It is a companion to the doc [Memory methods for continual learning: the big picture](https://claude.ai/artifact/JnG7owgYpsPUUfPsUhjnCr).*

Seventeen project ideas made the cut. Each fits in roughly 30–170 GPU-hours on one GPU for a full paper, and each gives a go/no-go answer within two weeks for about 20 GPU-hours or less. The three best starting points are listed below.

| Rank | Idea | Why start here |
| --- | --- | --- |
| #1 | Garbage collection for null-space knowledge editors | Fastest route to a crisp result: two days of CPU work, then about 10 GPU-hours |
| #2 | Banking and restoring the recurrent state of CUT3R/TTT3R across sessions | Best fit to the 3D section of the big-picture doc; needs no training |
| #3 | Keep only an "address" to a fact consolidated into a small LM | Tackles the doc's consolidation problem at the step where the store's copy can be deleted |

A web search for prior work rated every idea **partly novel**: a defensible new piece remains, but nearby pieces already exist. Each card below says which parts are taken and what is still new.

---

## At a glance

GPU-hours are the planning estimate for a full paper. "First signal" is the cost of the two-week go/no-go pilot. Tier A means start now, B means strong backup, and C means park the idea or fold it into another project.

| # | Idea | Memory store | Task / testbed | GPU-h (paper) | First signal | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [Garbage in the null space](#1-garbage-in-the-null-space-superseded-edits-block-fact-revision) | Weights | Sequential fact revision; GPT2-XL, phi-1.5 | ~150 | 2 CPU days + ~10 GPU-h | A |
| 2 | [Keep the state](#2-keep-the-state-bank-restore-and-warm-start-cut3rttt3r-state) | Fast state ↔ snapshot bank | Streaming 3D, cross-session revisits | ~120 | 15–20 GPU-h | A |
| 3 | [Keep the address, evict the text](#3-keep-the-address-evict-the-text) | Store → weights | Sequential fact injection, 0.5–1.5B LMs | ~150 | 10–15 GPU-h | A |
| 4 | [Rewrite or advance?](#4-rewrite-or-advance-compute-matched-replay-into-a-3d-recurrent-state) | Buffer → fast state | Streaming 3D at a fixed pass budget | ~150 | ~8 GPU-h | B |
| 5 | [Update frequency is not retention](#5-update-frequency-is-not-retention-auditing-multi-timescale-cms-memory) | Fast → slow levels | Theory, MQAR, HOPE audit | mostly CPU | CPU pilot already run (see `pilots/`) | B |
| 6 | [Displacement-aware experience memory](#6-displacement-aware-experience-memory) | External store (retrieval side) | ALFWorld/BabyAI agent streams | ~30 + API | ~3 GPU-h + $50–150 API | B |
| 7 | [Where should a frozen CUT3R keep a scene?](#7-where-should-a-frozen-cut3r-keep-a-scene-it-will-revisit) | All four, state → weights | Streaming 3D, cross-session revisits | ~160 | ~18 GPU-h | B |
| 8 | [Do-no-harm memory](#8-do-no-harm-memory-a-guarantee-against-memory-making-things-worse) | External store (text) | Reasoning streams with self-built memory | ~160 + API | 25–30 GPU-h | B |
| 9 | [Rent, buy, or let it expire](#9-rent-buy-or-let-it-expire) | Store → weights | Wikidata change stream, 0.5–1.5B LM | ~160 | ~15 GPU-h + CPU sim | B |
| 10 | [Overwrite or wait?](#10-overwrite-or-wait-bayesian-change-detection-for-memory-slots) | External store (slots) | Noisy fact updates in long conversations | ~120 + API | CPU week + ~25 GPU-h | B |
| 11 | [Folding session states into memory](#11-folding-session-states-into-persistent-memory) | Fast state → persistent state | Linear-attention/SSM LMs ≤1.5B | ~120 | ~10 GPU-h | B |
| 12 | [Revisits break persistence of importance](#12-revisits-break-persistence-of-importance-kv-eviction-audit) | KV cache | StreamVGGT/STream3R eviction | ~140 | ~15 GPU-h | C |
| 13 | [What do streaming 3D models remember?](#13-what-do-streaming-3d-models-remember) | Fast state / KV / pointer | Read-only memory probes | ~140 | ~8 GPU-h | C |
| 14 | [Beyond fact admission](#14-beyond-fact-admission-routing-by-type-of-change) | Cross-store router | Mixed stream of facts, skills, format shifts and noise | ~130 | ~20 GPU-h + $50 API | C |
| 15 | [Depth, not time](#15-depth-not-time-lineage-depth-and-memory-rot) | External store (insights) | Agent insight memory | ~170 | ~20 GPU-h | C |
| 16 | [How much replay does a pretrained ViT need?](#16-how-much-replay-does-a-pretrained-vit-need) | Data buffer | Class-incremental ViT | ~150 | ~8 GPU-h | C |
| 17 | [How budgeted 3D episodic memory forgets](#17-how-budgeted-3d-episodic-memory-forgets) | External store (3D) | Embodied QA across many scenes | ~60 + API | ~3 GPU-h + $30–100 API | C |

---

## Where to start

Run three short pilots in weeks 1–2. Together they cost under about 50 GPU-hours, and their results decide which idea becomes the lead project.

1. **#1, garbage in the null space.**
   - Days 1–2: a closed-form toy on CPU.
   - Then about 10 GPU-hours: GPT2-XL on 1,000 revision chains.
   - If it is a GO, it is the fastest route to a paper.
2. **#2, keep the state.**
   - An oracle-retrieval test on 7-Scenes revisit pairs, 15–20 GPU-hours.
   - Build the TTT3R fork and the revisit pairs so that #4 and #7 can reuse them unchanged.
   - Streaming 3D sees roughly one relevant arXiv paper a month, so start now.
3. **#3, keep the address.** Run this only if there is time.
   - A 10–15 GPU-hour pilot on Qwen2.5-0.5B, or Qwen3-0.6B to match O'Neill's setup.
   - It tests whether a short address cue brings back facts that are hidden in the weights.

At week 2, make #1 the first paper if it is a GO. Whichever of #2 or #3 passes its oracle test becomes the main line of work. The theory in #5 runs on CPU in the background at almost no cost.

## Four research arcs

Several ideas share code and data, so they group into four arcs. Each arc could become a set of related papers or thesis chapters.

| Arc | Ideas | Question | Shared infrastructure |
| --- | --- | --- | --- |
| **Streaming 3D as a four-store memory system** | #2 → #4 → #7, with #13 and #12 as diagnostic chapters | Is the CUT3R/TTT3R state a memory worth keeping (#2), refreshing (#4) or consolidating into weights (#7)? | One TTT3R fork with hooks for state dump, read-only decode and view injection; one revisit protocol on 7-Scenes, 12-Scenes and TUM |
| **Life cycle of a fact across store and weights** | #3, #1, #9, with #14 as a safety section | When does a fact move into the weights (#9)? What happens to the store's copy (#3)? What happens in the weights when the fact is revised (#1)? | Qwen2.5 0.5–1.5B and GPT2-XL; EasyEdit/AlphaEdit; CounterFact, zsRE, AToKe, WikiFactDiff |
| **Multi-timescale consolidation theory** | #5, #11, #10 | When does fast memory actually become slow memory, inside the model (#5, #11) and in an external store (#10)? | flash-linear-attention; delta-rule algebra; CPU simulations |
| **External-store hygiene for agents** | #6, #8, #15, #17 | How do you keep accumulated text memory from hurting? | API agents, Dynamic Cheatsheet/ACE builders, ALFWorld |

Arc 1 fits the 3D section of the big-picture doc best. Arc 3 needs the least compute and answers the doc's "main bet" on multi-timescale memory most directly.

---

## Idea cards

Each card names the specific lesson or failure mode in the big-picture doc that it targets. Full plans are in [`full-plans.md`](full-plans.md). They cover datasets, checkpoints, baselines, metrics, ablations and compute breakdowns.

### 1. Garbage in the null space: superseded edits block fact revision

*Weights · sequential fact revision (AToKe-ME revision chains, sLKE) · GPT2-XL and phi-1.5 with official AlphaEdit settings · ~150 GPU-h · Tier A*

**Idea.** Null-space and protected-set editors protect every past write: AlphaEdit, BetaEdit, OrthoEdit, EvoEdit and MEMIT_seq. When a fact is revised, the old write becomes garbage, but the editor keeps defending it. These editors compute the edit key from the subject alone, so a revision's key is almost identical to the key it replaces.
- With a soft Gram penalty, the revision is attenuated: after n earlier writes to the same subject, it takes about 1/(n+1) effect per layer.
- With a hard projector, the revision is almost fully blocked.
- Either way, a "ghost" of the old value stays protected.

The fix is a write ledger. It removes the stale key with a rank-1 downdate of the cached Gram (or a recomputed projector) and compacts periodically.

**Big-picture link.** This is the weights row of the doc ("protecting them costs plasticity"). It is also the weight-store version of "useful memories become faulty when continuously updated". The proposal is garbage collection for the weight store.

**First experiment (go/no-go).**
- Days 1–2, CPU only: a linear associative memory using AlphaEdit's exact closed form, to confirm the attenuation law and the downdate fix.
- Next, about 10 GPU-hours on GPT2-XL, comparing three edit sets:
  - 1,000 AToKe-ME chains with 3 revisions each;
  - 3,000 distinct-subject facts (same number of writes);
  - 1,000 final values edited once (same number of live facts).
- Four conditions: vanilla, +downdate, +rollback, +both.
- **GO** if all three hold:
  - revision chains trail the matched distinct facts by ≥10 points in current-value efficacy or ghost rate;
  - the realized fraction of each edit follows the prediction;
  - the downdate closes ≥50% of the gap, while rollback alone closes clearly less.

**Kill if** any of these holds:
- the gap at 3k writes is under 5 points;
- the realized fraction does not fall with prior same-subject writes;
- rollback alone fixes the gap as well as the downdate does.

**Closest work, and what is still new.**
- Already known:
  - same-subject key collisions: [S2RKE](https://arxiv.org/abs/2502.06868) and the [Covariance Trap](https://arxiv.org/abs/2603.15518);
  - a repeated-overwrite benchmark: [sLKE/MoKE](https://aclanthology.org/2025.acl-long.1492);
  - history-aware null-space editors: [BetaEdit](https://arxiv.org/abs/2605.09285), [OrthoEdit](https://aclanthology.org/2026.tacl-1.51.pdf) and [EvoEdit](https://arxiv.org/abs/2510.13851);
  - edit reversal: [2505.20819](https://arxiv.org/abs/2505.20819);
  - an [AlphaEdit reproducibility study](https://arxiv.org/abs/2606.26783) reporting degradation after about 5k edits.
- Still new: treating superseded writes as garbage in the protected set, the closed-form law for attenuation or blocking, the downdate ledger, and the claim that collapse scales with live facts rather than total writes. Neither the 12 agent searches nor an independent search found this.

**Main risk.** The margin in AlphaEdit's value optimization may absorb the attenuation, so binary efficacy barely moves. The fix itself is a one-line rank-1 downdate, so the paper must stand on the diagnosis and the scaling law.

### 2. Keep the state: bank, restore and warm-start CUT3R/TTT3R state

*Fast state ↔ external snapshot bank · streaming RGB 3D reconstruction · public CUT3R checkpoint with the TTT3R update rule · inference only, ~120 GPU-h on one 24–48 GB GPU · Tier A*

**Idea.** Treat the recurrent state as memory worth keeping.
- Copy state snapshots (about 1–2 MB in fp16) to a CPU bank, keyed by a place descriptor (DINOv2-SALAD).
- Before restoring a snapshot, verify it with a decode that does not update the state.
- Restore after a TTT3R reset or on a revisit.
- Warm-start a later session from the bank, so its poses come out directly in the first session's frame.
- Under a fixed bank budget, evict by coverage.

**Big-picture link.**
- It targets the fast-state failure "lost when the session ends unless consolidated".
- "Storage was never the real constraint" applies, since a snapshot is tiny.
- "The bottleneck shifts to memory access" becomes the design problem: how to retrieve and verify snapshots.

**First experiment (go/no-go).** An oracle-retrieval test over 2 weeks, 15–20 GPU-hours.
- **Cross-session**, on 7-Scenes plus two 12-Scenes rooms. Session 2 is decoded against the best session-1 snapshot. Compare it with:
  - a cold state fed that snapshot's keyframe;
  - a cold state fed the last 8 keyframes;
  - MASt3R + PnP.
- **Within-session**, with TTT3R reset every 100 frames on TUM and ScanNet loops. Compare:
  - pose-graph-only correction;
  - a hard switch to the saved state;
  - a per-token blend.
- **GO** if both hold:
  - the stored state beats the re-fed images by ≥5 points of recall at 25 cm / 10°;
  - restoring beats pose-graph-only by ≥10% ATE.

**Kill if** the stored state is no better than re-feeding its own keyframe images. The state would then be only a cache of its frames.

**Closest work, and what is still new.**
- Already exists:
  - [Scal3R](https://arxiv.org/abs/2609.04201): the same CUT3R backbone, SALAD/FAISS retrieval and GTSAM loop closure. It resets the state every 10 frames and re-injects archived pose tokens.
  - [ABot-Recon](https://arxiv.org/abs/2608.27529): loop closure as an optional backend.
  - [Anchor3R](https://arxiv.org/abs/2606.05035): finds that cached pose tokens fail when reused across coordinate frames.
  - Rules for overwriting and resetting the state: [TTSA3R](https://arxiv.org/abs/2601.22615), [MeMix](https://arxiv.org/abs/2603.15330), [ReCal3R](https://arxiv.org/abs/2607.05356) and [Info3R](https://arxiv.org/abs/2609.21938).
- Still new: restoring the full scene state, warm-starting a new session with release of changed regions, and a controlled test of stored state against stored images. An independent search found no work that saves CUT3R/TTT3R state across sessions.

**Main risk.** The state is tied to its coordinate frame. Blending states from differently drifted frames may cause pose jumps. Mitigations: a hard switch with re-anchoring, and verification before any restore.

### 3. Keep the address, evict the text

*Store → weights · sequential injection of invented facts, with a fixed 256-token retrieval context and hard distractors · Qwen2.5-0.5B/1.5B (Qwen3-0.6B to match prior work) · ~150 GPU-h · Tier A*

**Idea.** [O'Neill (Jul 2026)](https://arxiv.org/abs/2607.11020) found that facts written into a small LM's weights are mostly hidden by later writes rather than erased:
- the "forgotten" facts keep most of the log-probability their write added;
- wrong answers name recently written facts;
- putting the full fact text back in context restores 77–80% of them.

This suggests a third store state between "keep the text" and "evict": keep only a short address (a phrase, a 4-token code or a soft prompt) that re-cues the hidden copy in the weights. The text is deleted only when a delayed self-test, cued by the address, passes.

**Big-picture link.** This is the doc's open consolidation problem, at the step nobody measures: when the store can forget a fact once it is in the weights. Because old and new entries compete at retrieval ([2604.27003](https://www.emergentmind.com/papers/2604.27003)), short addresses also free up context.

**First experiment (go/no-go).**
- Setup: Qwen2.5-0.5B, 10 episodes × 200 facts.
- Test set: "hidden" facts, which fail with no cue but pass with the full text in context.
- Measure the share of the full-text gap each cue recovers:
  - the correct address;
  - a shuffled code;
  - a generic format prime;
  - an optimized generic suffix;
  - the address on the never-trained base model.
- **GO** if all hold:
  - the correct address recovers at least half the gap;
  - each control recovers 15% or less;
  - the eviction gate has under 20% regret at a 5× smaller store.

**Kill if** the address recovers less than 30% of the gap, or if a control recovers within 10 points of the address. Either means the effect is generic priming, not addressing.

**Closest work, and what is still new.**
- Already exists:
  - [Dual-Layer Agentic Memory](https://arxiv.org/abs/2608.22215) already consolidates store entries into weights by SFT and prunes the store, using a no-cue answerability gate (68% pruned at >98% EM).
  - [EVAF](https://arxiv.org/abs/2606.29916) gates consolidation with a test-retest protocol.
  - [TokMem](https://arxiv.org/abs/2510.00444) gives each memory item its own token.
- Still new: the address-only state, a delayed and cued eviction gate compared against the no-cue gate, and the economics per token of store and context under retrieval competition.

**Main risk.** O'Neill also reports slow relearning, which argues against facts being merely hidden. [REMIX](https://arxiv.org/abs/2411.07175) finds random keys are the most forgettable, so the natural-language address may beat the random code. That result is still publishable.

### 4. Rewrite or advance? Compute-matched replay into a 3D recurrent state

*Data buffer → fast state · streaming 3D at a fixed number of forward passes per second (ScanNet 3000-frame streams, TUM, KITTI) · CUT3R/TTT3R, no training · ~150 GPU-h on a 24 GB GPU · Tier B*

**Idea.** The model can afford only so many forward passes per second of video. A scheduler gives 5–50% of those passes to re-writing stored keyframes into the state instead of reading new frames.
- Replay writes are scaled by β.
- The pose memory is frozen during replays.

The headline is an accuracy-vs-passes frontier against every equal-compute alternative: a tuned frame stride, adaptive keyframes, and AFG/Info3R gating. Replay is also stacked on TTSA3R/ReCal3R. The gap between a keyframe's stored pose and its re-predicted pose is a free forgetting signal for adapting the replay rate.

**Big-picture link.** "Replay is still the bar to beat, and compute is the budget that matters", applied to a fast state.

**First experiment (go/no-go).**
- About 8 GPU-hours.
- A ~100-line scheduler, run on 16 sequences in about 40 configurations.
- **GO** if all three hold:
  - some replay with β > 0 beats the tuned stride by ≥10% median ATE on sequences of ≥1000 frames;
  - that gain also appears on exploration sequences, not only loops;
  - β > 0 beats β = 0.

**Kill if** any of these holds:
- no replay setting gains ≥5% over the tuned stride;
- the gains appear only on loops, which would make this the same as Anchor3R/ABot-Recon re-insertion;
- the gains drop below 5% once stacked on ReCal3R or TTSA3R.

**Closest work, and what is still new.**
- Already exists:
  - CUT3R's offline revisit mode (twice the compute, not online);
  - [Anchor3R](https://arxiv.org/abs/2606.05035) (offline re-insertion of loop keyframes);
  - [AFG](https://arxiv.org/abs/2605.16981) and [Info3R](https://arxiv.org/abs/2609.21938), which change only how the newest frame writes.
- Still new: online replay that writes into the state under a fixed pass budget.

**Main risk.** Replay may turn out to be just a smarter keyframe stride.

### 5. Update frequency is not retention: auditing multi-timescale (CMS) memory

*Fast → slow levels inside the model · theory, linear simulations, DeltaNet with two slow MLP levels on MQAR, and an audit of a HOPE reimplementation · mostly CPU, ≤10 GPU-h for the first nonlinear test · Tier B*

**Idea.** Nested Learning/HOPE's continuum memory system (CMS) assumes that a level updated less often acts as slower memory. Suppose all levels follow one shared loss with plain SGD and no per-level decay or reset. Then each slower level is exactly a lagged, scaled copy of the fastest level, W_k = (η_k/η_1)·W_1 at the last boundary, so it carries nothing extra across a session reset.

Fast-level decay or periodic downscaling (W_fast ← (1−γ)·W_fast every S steps) breaks this degeneracy. The project derives a load-aware schedule for it.

**Big-picture link.** This tests the doc's "main bet", memory on several timescales, and its fast-state failure, "lost when the session ends".

**Pilot already run (CPU, in [`pilots/cms-degeneracy/`](../pilots/cms-degeneracy/)).**
- The lemma holds to 3e-15.
- Under plain shared-loss CMS, the slow level's recall coefficient falls from 0.024 to 0.014 as the spacing gap grows, against 0.077 for a slow-only learner.
- With fast-level decay of 0.97 per step, an optimum appears: 0.067 at gap 128, 2.8× the plain CMS value.

**Next.**
- Week 1: parameter sweeps and closed forms.
- Week 2: DeltaNet plus 2 slow MLP levels on Zoology MQAR.
- **GO** if both hold:
  - the theory fits (R² ≥ 0.9);
  - in the nonlinear model, plain CMS keeps ≤50% of slow-only recall after a reset, while the derived schedule gives ≥1.5×.

**Kill if** nonlinear CMS already keeps ≥80% of slow-only recall. The degeneracy would then be a linear artifact.

**Closest work, and what is still new.**
- The general principle is prior art and must be credited:
  - [Jones et al., ICLR 2023](https://iclr.cc/virtual/2023/poster/10732) (multiscale learners);
  - Smith et al. 2006 (two-state motor learning);
  - [Benna & Fusi 2016](https://arxiv.org/abs/1507.07580) (cascade synapses).
- [Language Models Need Sleep (Behrouz et al., 2606.03979)](https://arxiv.org/abs/2606.03979) adds explicit distillation for consolidation to Nested Learning, which is indirect support for the premise.
- Still new: the CMS-specific audit, the discrete-period and delta-rule load terms, the session-reset metric, and the derived schedule. No paper was found showing that CMS levels are lagged copies.

**Main risk.** The lemma looks obvious once stated. Real HOPE uses chained MLPs and Adam/Muon, which break exact proportionality, so the nonlinear audit carries the paper.

### 6. Displacement-aware experience memory

*External store, retrieval side · sequential ALFWorld/BabyAI experience reuse (the [2604.27003](https://www.emergentmind.com/papers/2604.27003) protocol) · frozen API agent and small embedders · ~30 GPU-h plus API · Tier B*

**Idea.** This is a training-free study of retrieval-side forgetting. Newer experiences push older ones out of a capped retrieval budget, and that may explain most of the backward-transfer loss. Two fixes are tested:
- contrastive key rewriting, accepted only if the probe margin rises for both confusable entries;
- a one-slot-per-collision-group cap.

Plain "make each key retrieve itself" optimization serves as a foil. The [Crowded Embedding Space](https://arxiv.org/abs/2606.28343) theory predicts it makes crowding worse.

**Big-picture link.** This targets "with a finite context, old and new experiences compete at retrieval, so the bottleneck shifts to memory access".

**First experiment (go/no-go).**
- About 3 GPU-hours plus $50–150 of API.
- Reproduce the ALFWorld stream at k=3 entries.
- Run an oracle that removes only newer competitors from old-task queries, and see how much backward transfer it recovers.
- **GO** if all hold:
  - the oracle recovers ≥50% of the backward-transfer gap, or ≥4 points;
  - the method gets ≥40% of that headroom;
  - the method gains ≥3 points over MMR/diverse retrieval at equal tokens.

**Kill if** the oracle recovers under 3 points. Forgetting would then come from the reader, not from slot competition.

**Closest work, and what is still new.**
- [REALM](https://arxiv.org/abs/2609.16053) does memory reconsolidation; its full method was not readable.
- [Entity-Collision](https://arxiv.org/abs/2605.29630) is a protocol for attributing retrieval lift.
- Still new: measuring displacement as the cause of backward-transfer loss, and margin-gated contrastive rewriting.

**Main risk.** Earlier work found that write strategy matters much less than retrieval, so the effect may be small.

### 7. Where should a frozen CUT3R keep a scene it will revisit?

*All four stores, state → weights · cross-session revisits on 7-Scenes, 12-Scenes and TUM · ~160 GPU-h (the judge rated it compute-heavy) · Tier B*

**Idea.** A frozen CUT3R/TTT3R streams visit A of a scene, then visit B. What A taught is kept in one of five ways, all at about 1–2 MB and ≤60 s:
- carried state;
- a retrieved snapshot;
- replayed keyframes;
- a per-scene LoRA, distilled with the Free Geometry recipe;
- an ACE-style scene-coordinate head.

The study asks which wins as A grows. It also asks whether a forgetting score computed at the end of A without ground truth predicts when the adapter is worth storing.

**Big-picture link.** This is the doc's consolidation question inside its own 3D section: when has a scene earned the weights?

**First experiment (go/no-go).**
- About 18 GPU-hours on 14 revisit pairs from 7-Scenes.
- **GO** if all three hold:
  - at length 1000, the adapter beats the best non-weight store by ≥5% ATE or accuracy, with a CI excluding 0;
  - the gain grows with the length of A;
  - it beats both test-time adaptation run on B itself and an adapter built for a different scene.

**Kill if** test-time adaptation on B matches the result. The gain would then be adaptation, which is already published, not memory.

**Closest work, and what is still new.**
- Already exists:
  - [Free Geometry](https://arxiv.org/abs/2604.14048): test-time LoRA distillation;
  - [LoRA3D](https://arxiv.org/abs/2412.07746): per-scene adapters;
  - [ACE-G](https://arxiv.org/abs/2510.11605): per-scene weights for relocalization.
- Still new: a cross-session protocol for streaming reconstruction foundation models, a comparison of stores at matched bytes and compute, and a signal for when a scene has earned weights.
- It shares all its infrastructure with #2 and #4.

**Main risk.** It is effectively a benchmark paper, and Free Geometry's average gains are only about 3–4%.

### 8. Do-no-harm memory: a guarantee against memory making things worse

*External store (consolidated text memory) · test-time learning streams (Game of 24, MATH level 5, MMLU-Pro Physics, an ARC-AGI-1 subset) · Qwen3-8B on vLLM plus one black-box API model · ~160 GPU-h plus API · Tier B*

**Idea.** A read gate, calibrated with conformal risk control on paired with-memory and no-memory outcomes, so that memory breaks problems the model already solves at most α of the time.
- Adaptive conformal inference recalibrates the gate online as the memory drifts.
- A teacher-forced "anchor-NLL" write gate replaces costly rollout-based consolidation gates.

**Big-picture link.** [2605.12978](https://arxiv.org/abs/2605.12978) found that text memory consolidated by an LLM improves, plateaus, then falls below the no-memory baseline. This idea bounds that damage.

**First experiment (go/no-go).**
- 25–30 GPU-hours.
- **GO** if all three hold:
  - paired harm is ≥5 points above the sampling-noise floor late in the stream;
  - harm can be predicted (AUROC ≥ 0.70);
  - the conformal gate meets α while keeping ≥50% of the memory gain.

**Kill if** harm is under 2 points above the noise floor (nothing to control), or harm cannot be predicted (AUROC < 0.62).

**Closest work, and what is still new.**
- Rollout-based write gates: [Janus](https://arxiv.org/abs/2606.31121), [Recuris](https://arxiv.org/abs/2608.24876) and [AGMR](https://arxiv.org/abs/2607.17621).
- Heuristic read gates: [RSCB-MC](https://arxiv.org/abs/2604.27283) and [MemTrim](https://arxiv.org/abs/2610.07311).
- Measurement of harm: [Memory Trust Gap](https://arxiv.org/abs/2609.01852).
- Still new: a finite-sample and online guarantee of harm relative to no memory, which works for black-box API models.

### 9. Rent, buy, or let it expire

*Store → weights · keeping a 0.5–1.5B LM current over 24 monthly Wikidata change steps, with pageview-driven demand · ~160 GPU-h (the judge flagged this as optimistic) · Tier B*

**Idea.** Moving a changing fact from RAG into the weights is framed as an online rent-or-buy problem with two twists.
- A purchase expires when the fact changes, at a per-relation hazard rate.
- Buying a fact shrinks the store, which lowers the retrieval cost of every other fact.

The buy price is a forecast, made before writing, of the KL drift the write will cause. The policy is scored against the clairvoyant schedule.

**Big-picture link.** "Forgetting tracks how far an update moves the model" becomes a price. "Retrieval competition" becomes a rent that grows with store size.

**First experiment (go/no-go).**
- About 15 GPU-hours to forecast realized KL for ~500 CounterFact edits on Qwen2.5-0.5B, plus a synthetic world on CPU.
- **GO** if all three hold:
  - the forecast reaches Spearman ≥ 0.4;
  - realized prices vary at least 3× between the 25th and 75th percentiles;
  - in simulation the policy stays within 2× of the optimum.

**Kill if** the forecast fails (the project becomes theory plus simulation), or if buying never pays off at small scale.

**Closest work, and what is still new.**
- [ALLOT](https://arxiv.org/abs/2609.32344) does budgeted static routing of updates to weights.
- [Dual-Layer](https://arxiv.org/abs/2608.22215) does write routing with periodic write-back.
- Still new: expiring purchases, the store-size externality, and the competitive ratio against a clairvoyant schedule over time.

### 10. Overwrite or wait? Bayesian change detection for memory slots

*External store (structured slots) · long-term conversational memory with hedged, hearsay, retracted and temporary updates (MemOps recency_trap, MemoryAgentBench, LongMemEval-KU) · ~120 GPU-h plus API · Tier B*

**Idea.** Each memory slot runs Bayesian online change-point detection.
- The speaker's stance (asserted, hedged, hearsay or retracted) sets the observation noise.
- The attribute class sets the hazard.
- Latest-wins is the special case with zero noise.
- Temporary scopes ("in Lyon this week") open sub-states that expire.

A theory part on CPU shows that coupled fast/slow memories like Memini's entrench facts that were repeated often.

**Big-picture link.** External memory moves the stability–plasticity dilemma to the overwrite decision. This also connects to the doc's multi-timescale bet (Memini).

**First experiment (go/no-go).**
- Week 1: filter simulations on CPU.
- Week 2: about 25 GPU-hours with Qwen2.5-7B extracting the updates.
- **GO** if all three hold:
  - the filter beats latest-wins by ≥10 points on noisy updates and halves false updates;
  - it stays within 2 points on clean data;
  - LLM extraction keeps ≥60% of the oracle-extraction gain.

**Kill if** the filter does not beat a tuned "confirm k times" rule by ≥5 points.

**Closest work, and what is still new.** Prior work includes [BeliefMem](https://arxiv.org/abs/2605.05583), [MemOps](https://arxiv.org/abs/2607.12893), [MINTEval](https://arxiv.org/abs/2605.18565) and [Memini](https://arxiv.org/abs/2605.05097). The judge noted that the entrenchment "law" is close to a definition for a linear filter.

### 11. Folding session states into persistent memory

*Fast state → persistent state · MQAR-style sessions and a CounterFact track on RWKV-7 1.5B, DeltaNet 1.3B, GLA and Mamba2 · ~120 GPU-h · Tier B*

**Idea.** Fold 20 sessions' end states into one persistent initial state or read term, with no gradients. Each head solves a RegMean-style ridge problem over the query Grams it computed on its own session. Two extra terms shape it:
- a reference Gram computed on generic text, which protects generic behaviour;
- recency weighting, which resolves contradictions between sessions.

The project then maps where folding works, by decay, state rank and query/key alignment.

**Big-picture link.** Fast state is "lost when the session ends unless consolidated". This tests one cheap way to consolidate it.

**First experiment (go/no-go).**
- About 10 GPU-hours.
- **GO** if, at 10 sessions, all three hold:
  - the fold recovers ≥50% of in-context recall;
  - it beats State Soup averaging by ≥10 points;
  - generic KL stays ≤ 0.02.

**Kill if** any of these holds:
- recall is under 30%;
- gradient state tuning wins at equal compute;
- PRECOG-style composition matches the fold.

**Closest work, and what is still new.**
- Already exists:
  - [State Soup](https://arxiv.org/abs/2406.08423) and [PRECOG/SMC](https://arxiv.org/abs/2608.02560) already persist sessions by averaging or composing states;
  - [FAAST](https://arxiv.org/abs/2605.04651) does closed-form fast weights;
  - [Lee et al.](https://arxiv.org/abs/2605.26099) do sleep-time consolidation with backprop.
- Still new: the protected ridge merge itself, refitting the queries bottom-up, and the diagnostic map of where folding works.
- The judge scored novelty 2/5.

### 12. Revisits break persistence of importance: KV eviction audit

*KV cache · StreamVGGT/STream3R on ScanNet loops, TUM and stitched 7-Scenes A-B-A streams · ~140 GPU-h plus 1–2 TB of attention logs (the judge said this is understated) · Tier C*

**Idea.** An oracle audit of KV eviction for streaming 3D transformers.
- A closed-loop oracle that knows future attention sets a reference for every published eviction policy.
- Results are split by revisit type: novel frames, short-gap revisits and long-gap revisits, defined by ground-truth view overlap.
- It tests the assumption that tokens important now stay important later.
- A place-quota buffer serves as the minimal continual-learning fix.

**Closest work.** [RetrieveVGGT](https://arxiv.org/abs/2605.09644) and [STAC](https://arxiv.org/abs/2603.20284) already do place-aware eviction, so the fix alone is not new. The audit and the stratification are new. This works best as a diagnostic chapter of Arc 1.

### 13. What do streaming 3D models remember?

*Fast state / KV cache / pointer memory · read-only probes on 7-Scenes, NRGBD, TUM, ScanNet and 3RScan rescans · ~140 GPU-h · Tier C*

**Idea.**
- Continual-learning accuracy matrices for streaming 3D memories, filled by probes that do not write to memory.
- A-B-A scene switches and 3RScan rescans measure plasticity.
- A delta-rule model of TTT3R predicts the best reset period.

**Closest work.** A gate-derived horizon of about 3 frames ([2605.16981](https://arxiv.org/abs/2605.16981)), [HorizonStream](https://arxiv.org/abs/2605.23889) and [ABot-Recon](https://arxiv.org/abs/2608.27529) cover much of the framing. This works best as the measurement chapter for #2 and #4: the probes come almost free from the same runs.

### 14. Beyond fact admission: routing by type of change

*Cross-store router · one mixed stream of new facts, revised facts, new skills, format shifts and noise · Qwen2.5-0.5B/1.5B with LoRA · ~130 GPU-h · Tier C*

**Idea.** [Harrington et al. (2607.07847)](https://arxiv.org/abs/2607.07847) found that the kind of change decides whether adaptation needs the weights. This project routes each update by type to one of four places, using cheap signals:
- a context rule;
- the external store;
- a LoRA;
- nowhere (drop it).

Any weight write must first be corroborated.

**Closest work.** [Dual-Layer](https://arxiv.org/abs/2608.22215) already routes fact writes and consolidates them, and [D-MEM](https://arxiv.org/abs/2603.14597) does surprise-gated routing. What remains is incremental: updates that are not facts, and the corroboration gate.

### 15. Depth, not time: lineage depth and memory rot

*External store (consolidated insight memory) · a new CPU-only "Hidden-Rule World", ALFWorld, ARC and Game of 24 · ~170 GPU-h · Tier C*

**Idea.**
- Each insight records how many LLM rewrite passes separate it from the raw episodes.
- That depth, not elapsed time, should predict when rewritten memory rots.
- The proposed fixes are depth-bounded regeneration from the raw episodes, or tree consolidation with O(log N) depth.

**Closest work.** [2605.12978](https://arxiv.org/abs/2605.12978), [2609.25052](https://arxiv.org/abs/2609.25052) and [MemLineage](https://arxiv.org/abs/2605.14421). The judge called the pre-registered prediction of the accuracy peak optimistic.

### 16. How much replay does a pretrained ViT need?

*Data buffer · class-incremental ImageNet-R and CIFAR-100, comparing ViT LoRA, ViT full fine-tuning and ResNet-18 from scratch (Mammoth) · ~150 GPU-h, first signal ~8 GPU-h · Tier C*

**Idea.** Measure r*(T, ε), the smallest replay fraction that keeps forgetting under ε, as it grows with the number of tasks T, separately for each size of trainable parameter set. Then race three predictors of per-class forgetting:
- KL on old data against the logits stored at write time;
- KL on new-task data (RL's Razor);
- distance in parameter space.

**Closest work.**
- Cho et al. is now titled [Forget Forgetting (ICLR 2026)](https://arxiv.org/abs/2502.07274).
- [Bethune et al.](https://arxiv.org/abs/2502.06042) fit scaling laws of forgetting against the replay fraction for LLMs.
- [2602.22479](https://arxiv.org/abs/2602.22479) already runs a PI replay controller.
- Still new: the task-count and trainable-capacity axes in vision, and the race between predictors.

### 17. How budgeted 3D episodic memory forgets

*External store (3D scene memory) · OpenEQA across 1–16+ scenes under a fixed memory budget · frozen perception models ≤1.5B plus an API VLM · ~60 GPU-h plus API · Tier C*

**Idea.** Cache the RGB-D perception stream once, then replay it through many memory policies at the same budget. Three comparisons:
- geometry-grounded merging against LLM-rewrite consolidation;
- whether a set-cover model predicts where accuracy collapses;
- whether a recall proxy that needs no queries can replace VLM evaluation.

**Closest work.** The area is crowded: [LT-Mem](https://arxiv.org/abs/2608.19059), [eMEM](https://arxiv.org/abs/2606.03374), [EMem-Bench](https://arxiv.org/abs/2609.28236) and [Sequential EQA](https://arxiv.org/abs/2607.21571).

---

## Ideas considered and dropped

The brainstorm produced 48 ideas. Thirteen went through prior-work checking and planning, and four more were added later for angles the first round missed. Some ideas were merged into the shortlist. Those dropped outright include:

| Dropped idea | Main reason |
| --- | --- |
| Ski-rental consolidation of agent text memory | Duplicates #9 |
| Sharded consolidation for exact deletion | Weakest tie to the doc's lessons; Agentic Unlearning may already cover it |
| Buffer as a write-location oracle for CLIP neuron masks | Crowded field (SPU, MIST, GNSP); weak signal from polysemantic neurons |
| kNN store + slow head for online geolocation | Small headroom over ACM; question already covered by #9 |
| Rewritten replay targets as iterated self-distillation | Overlaps X-DER |
| Token-budgeted, fidelity-allocated ViT replay | An efficiency trick rather than memory management |
| Forward-only buffer-calibrated adapter merging | Scoop risk from adapter-merging work |
| Spaced, collision-targeted replay of facts in weights | Crowded: MIR, FOREVER, SRT |
| Read-twice selective rehearsal for recurrent LMs | Close to JRT; a BM25 replay baseline may dominate |
| Versioned spill memory for overwritten delta-rule associations | The new readout path is out of distribution for pretrained models |
| KV compression in hybrid SSM-attention models | Closer to inference efficiency than to continual learning |
| Benna–Fusi cascade state for CUT3R | Risks out-of-distribution states; diagnostic folded into #13 |
| Complementary replay of what retrieval cannot rescue | In-context gains at 1.5B may be near zero |
| Pinned core memory vs retrieval tiering | Close to Letta core memory and PPRO |
| Corroboration-gated promotion via GRPO | The most compute-heavy idea; the gate is folded into #14 |

The full list of 48 is in [`data/pass1_result.json`](data/pass1_result.json), under `generated` and `dropped`.

## How this was made, and how far to trust it

**Process.**
- First round: eight brainstorming agents, each covering one angle (streaming 3D, agent memory, replay under a compute budget, small LMs, fast state, multi-timescale theory, routing between stores, and benchmarks), produced 48 ideas.
- A selection step kept 13. Each of the 13 got an adversarial web search for prior work, then a planning pass that repositioned it around whatever remained new.
- A judge then ranked the 13.
- Second round: a live re-check of the 5 ideas whose first prior-work search had failed, plus 4 new ideas for gaps the judge found. All 17 were then re-ranked together.

**Limits.**
- **Prior-work checks used search snippets, not full papers.** arXiv, OpenReview and Semantic Scholar pages could not be opened from this environment, so every verdict rests on search-result text.
  - Before committing to an idea, read the closest-work papers on its card in full.
  - Each plan names the papers to read first.
- **arXiv IDs come from search results, not from opened pages.** Two corrections from checking:
  - Harrington et al. (2607.07847) was confirmed to exist after one agent wrongly failed to find it.
  - Cho et al. (2502.07274) is now titled *Forget Forgetting: Continual Learning in a World of Abundant Memory* (ICLR 2026), so the big-picture doc's Sources line may need updating.
- **GPU-hour figures are planning estimates.**
  - The judge flagged #9, #12, #15 and #8 as optimistic.
  - #5's 130 GPU-hours is too high, since most of it runs on CPU.
- **These areas move fast.** Several 3D ideas compete with papers posted in the last three months. Post early results quickly.
- Model names, checkpoints and API prices in the full plans should be checked before budgeting.
