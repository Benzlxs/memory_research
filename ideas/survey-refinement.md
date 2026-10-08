# Refining the shortlist with the agent-memory survey

*2026-10-08. This refines [memory-management-ideas.md](memory-management-ideas.md) (v1) using Hu et al., [Memory in the Age of AI Agents: A Survey](https://arxiv.org/abs/2512.13564) (Dec 2025, revised Jan 2026). Throughout, "#N" always means card N in v1. The refined cards are labelled R1–R8 by new rank, and each heading gives both labels.*

The survey sharpens framing and supplies baselines, but it adds no novelty, so every v1 prior-work verdict still stands. Its biggest effect is that several cards can now be framed as tests of claims the survey makes, which strengthens them.

The main changes from v1:

| Change | Cards | Detail |
| --- | --- | --- |
| Portfolio shrinks | 17 → 13 standing projects, plus 4 new ideas | #7 and #13 merge into #2; #14 merges into #9; #16 is dropped |
| Top two stay in Tier A | #1, #2 | Both get sharper headlines in the survey's terms |
| #3 leaves Tier A | #3 | Survey-listed rivals and three design flaws |
| #8 moves to the top of Tier B | #8 | Rescoped to 25–40 GPU-hours as a guarantee that works with any memory builder |
| New Tier B idea | "Ghosts in the pointer memory" | Stale 3D memory after the world changes |
| Pilots lead with a survey claim to falsify | #1, #2, #5, #8, #10, Ghosts | The timing claim is shared by several cards, but only #5 makes it the headline (see the claims table) |
| Survey-listed baselines become mandatory | Most cards | Each could kill a card |
| Compute | All cards | Stays within ≤1.5B models; the A+B pilots cost about 60–70 GPU-hours in total |

---

## How the survey was read

arXiv and every PDF mirror are blocked from this environment. The survey was read through:
- the authors' companion repository ([Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)): its two overview figures and its list of about 200 papers, sorted into the survey's own grid;
- the abstract and table-of-contents fragments;
- search snippets quoting the text.

[`survey-digest.md`](survey-digest.md) records what came from where. One quote matters to several cards and comes only from a snippet: *short- and long-term memory "emerge from the temporal patterns with which formation, evolution and retrieval are engaged"*. **Check it against the PDF before quoting it in a paper.** If you can drop the PDF into the repo, I will re-check every claim marked "snippet".

## The survey's map

The survey sorts agent memory along three axes:
- **Form:** what carries the memory — token-level text, parametric weights, or latent hidden states.
- **Function:** what it is for — factual, experiential or working memory.
- **Dynamics:** three lifecycle operators — formation (F), evolution (E = consolidation ∪ updating ∪ forgetting) and retrieval (R).

Its paper list fills the grid very unevenly:

| Function \ Form | Token-level | Parametric | Latent |
| --- | ---: | ---: | ---: |
| Factual | 84 | 16 | 8 |
| Experiential | 46 | 6 | 1 |
| Working | 14 | 2 | 22 |

Three things follow for this portfolio.
- **Most strong ideas move memory between cells.** The survey covers such moves as formation routes (parametric internalization, latent representation) and lists MAC and SELF-PARAM for text → weights. What it does not treat as a management decision is *when* to move, and what happens to the source copy afterwards. Examples:
  - #2 promotes 3D recurrent state from working to factual memory;
  - #3 and #9 move facts from token-level to parametric;
  - #5 moves fast state to slow weights.
- **Streaming 3D reconstruction is absent from the paper list.** The abstract names multimodal memory as a frontier; world-model memory appears only in a third-party summary. The closest entries are XMem, MemoryVLA, Mem2Ego and Context-as-Memory. Like the sparse cells, this absence may reflect scope rather than open ground, so the 3D arc should be pitched on 3D baselines, not on filling a survey gap.
- **A sparse cell reflects the survey's scope, not open ground.** The list stops around January 2026 and misses every 2025–26 direct competitor found in v1 (BetaEdit, Dual-Layer, Scal3R, Free Geometry).

| Idea | Form move | Function move | Main operator |
| --- | --- | --- | --- |
| #1 | parametric | factual | E: updating / forgetting |
| #2 | latent → latent repository | working → factual | R: timing; E: updating |
| #3 | token-level → parametric | factual | E: forgetting (cross-form delete) |
| #4 | token buffer → latent | working | E: consolidation, under a compute budget |
| #5 | latent → parametric | working → factual | E: consolidation vs forgetting |
| #8 | token-level | experiential | R: timing (when to inject) |
| #10 | token-level | factual | E: updating |
| Ghosts (new) | latent | factual | E: updating / forgetting |

---

## Refined portfolio

| Rank | Idea (refined) | v1 | Tier | Survey claim it tests | Pilot |
| --- | --- | --- | --- | --- | --- |
| R1 | [Parametric memory without delete](#r1--parametric-memory-without-delete-v1-1) | #1 | A | Model editing is the "updating" operator for parametric memory | 2 CPU days + ~10 GPU-h |
| R2 | [More than a cache?](#r2--more-than-a-cache-v1-2) (absorbs #7, #13) | #2 | A | §7.1 retrieval → generation; the third-party critique that latent memory is "just caching" (timing claim secondary) | ~10–12 GPU-h |
| R3 | [Certified retrieval timing](#r3--certified-retrieval-timing-v1-8) | #8 | B (top) | Memory management should move to learned/RL policies (§7.2–7.3) | ≤5 GPU-h + API |
| R4 | [Forgetting, not frequency](#r4--forgetting-not-frequency-v1-5) | #5 | B | Short vs long term comes from the timing of F/E/R (lead test) | ~8–10 GPU-h |
| R5 | [Overwrite or wait? A closed-form threshold](#r5--overwrite-or-wait-a-closed-form-threshold-v1-10) | #10 | B | Survey-listed RGMem's dataset-invariant threshold; learned vs derived gates | CPU + ~$50–100 API |
| R6 | [Ghosts in the pointer memory](#r6--ghosts-in-the-pointer-memory-new) | new | B | Survey-listed Scissorhands' "persistence of importance" | ~6 GPU-h + 1.5 weeks engineering |
| R7 | [Keep the index, evict the text](#r7--keep-the-index-evict-the-text-v1-3) | #3 | B (was A) | §7.1, from retrieval to generation | ~10–15 GPU-h |
| R8 | [Rehearse or advance?](#r8--rehearse-or-advance-v1-4) | #4 | B | Timing claim, secondary to #5; main test is replay vs reset-and-rebuild | ~10 GPU-h |
| 9–17 | [Tier C and merged ideas](#tier-c-merged-and-dropped) | | C | | |

## Where to start

**Days 1–3: reading gates, no GPU.**
- Get the survey PDF and check the "temporal patterns" quote and Section 7.
- #1: read BetaEdit's analysis section and the AlphaEdit reproducibility study ([2606.26783](https://arxiv.org/abs/2606.26783)).
- #5: check whether the HOPE reimplementation keeps Titans' forget gate and momentum.
- #10: check what RGMem's threshold counts, in its code.
- #2: confirm CUT3R's read-only raymap path and its state size in code.

**Weeks 1–2: three pilots, under about 35 GPU-hours.**
1. #1: a CPU toy, then BetaEdit with and without stale-key removal on GPT2-XL revision chains, then AlphaEdit.
2. #2: the cross-session cache test on 7-Scenes and 12-Scenes. This also builds the shared TTT3R fork.
3. #5: the 2×2 forgetting × frequency test on MQAR, in the background.

**Week 3, if budget remains:** a pilot of #8 (certified retrieval timing) using an API model, about 5 GPU-hours plus API.

**Decision rules (thresholds are in each card):**
- **#1:** a GO makes it paper 1. If it is killed, #8 becomes the fast paper, but only if #8's pilot passes; if that pilot has not run, run it next.
- **#5:** if the forgetting effect is ≥3× the frequency effect, Arc 3 continues as a CPU-heavy side paper. If frequency adds ≥25% relative recall beyond forgetting, publish the frequency-positive result. Anything in between is inconclusive, and Arc 3 is parked.
- **#2 GO:** Arc 1 becomes the main line. Start the Ghosts gate on 3RScan rescans: about 6 GPU-hours plus about 1.5 weeks of engineering on a Point3R harness. #2's TTT3R fork cannot delete individual memory entries, so only its data loaders and evaluation code carry over.
- **#2 killed:** run the Ghosts gate anyway, and schedule #3's 0.6B pilot (about 10–15 GPU-hours) in week 4. That pilot decides whether Arc 2 grows beyond #1.

## Research arcs

| Arc | Cards | Question | Survey hook |
| --- | --- | --- | --- |
| **1. Streaming 3D as a memory system** (main line if #2 passes) | #2 (with #7 as phase 2), #4, Ghosts; #12 and the start-state pool as extras | Is the recurrent 3D state memory or a cache? Can rehearsal extend it? What happens when the world changes? | Streaming 3D is absent from the survey's paper list; multimodal memory is a stated frontier |
| **2. Life cycle of a fact across store and weights** | #1 (lead), #3, #9 | Weights have no DELETE; when can the store's copy go; when should a changing fact enter the weights? | The token–parametric synchronization gap that critics of the survey point out |
| **3. Timescale theory** (cheapest) | #5, #10, #11 | What actually turns fast memory into long-term memory? | The survey's central short-term/long-term claim |
| **4. Trustworthy agent memory** (secondary, API-heavy) | #8, #6, #15, RL-manager audit, #17 | How do you keep accumulated memory from making things worse? | §7.2–7.3 and the trustworthy-memory frontier |

---

## Refined cards (Tiers A and B)

### R1 · Parametric memory without delete (v1 #1)

*Tier A · factual × parametric · ~150 GPU-h on a 24–48 GB card*

**What the survey adds.**
- The survey files model editing as the "updating" operator for parametric memory.
- This card predicts that protected-set editors (AlphaEdit's soft penalty; BetaEdit, OrthoEdit and EvoEdit's hard projectors) only ever add. A real update needs an explicit forgetting step.
- Text stores already have add/update/delete (Mem0, Memory-R1) and Zep-style invalidation. The survey's factual × parametric cell has only add.
- The ledger is the missing operator that keeps a weight store in sync with the text store.

*Terms:*
- *Protected set:* the keys of past edits whose outputs the editor keeps fixed.
- *Hard projector:* BetaEdit/OrthoEdit-style projection of each update onto the null space of those keys.
- *Downdate (stale-key removal):* a rank-1 removal of a superseded edit's key from the protected set.
- *Garbage collection:* applying the downdate whenever a fact is revised.
- *Stale-value rate* (v1's "ghost rate"): the share of revised facts whose top answer is still a superseded, non-original value.

**Changes from v1.**
- **The main figure becomes a 2×2.**
  - Rows: an *append*, where the old value is still true, versus a *supersession*, where the old value is now false. Appends use the PEAK benchmark from the survey-listed [Neighboring Perturbations of Knowledge Editing](https://openreview.net/forum?id=K9NTPRvVRI), run on GPT-2 XL.
  - Columns: protection kept versus downdated.
  - Prediction: downdating helps supersession and hurts appends, so garbage collection must check whether the relation is one-to-one.
  - If downdating does not hurt appends, pivot to "protecting edited keys is unnecessary", which is still a publishable negative result.
- **Add "correcting a wrong edit" as a headline use case,** for the trustworthy-memory frontier. It uses AToKe's 73 A→B→A chains and CounterFact edit-then-revert pairs.
- **Keep history as text, not weights.** Superseded values become dated records in the ledger, invalidated Zep-style. Time-qualified questions are answered by putting the record in context. Compare with PRISM Edit.
- **Add a key-design ablation** (last-token keys instead of subject-only keys). It separates address collision from garbage and pre-empts "just fix the key".
- **Run BetaEdit first.** Near-total blocking under a hard projector is the larger effect, and it is less likely to be hidden by AlphaEdit's value-optimization margin.
- **Add two brute-force rows:**
  - periodic compaction with cached targets, as an accuracy-vs-compute frontier;
  - the ledger's current value placed in context. This forces the paper to say why fixing the weights matters at all (closed-book use, on-device use, or consolidation).
- **Drop the Llama3-8B point.** Use GPT2-XL and phi-1.5 only.
- **Correct the WISE description:** it is a sharded side memory, so measure it rather than assume it is immune.

**Pilot.**
- A CPU toy first.
- Then about 10 GPU-h on GPT2-XL with BetaEdit, with and without stale-key removal. Edit sets: 1,000 AToKe-ME chains (3 revisions each), matched distinct-subject facts, and A→B→A reversions.
- **GO** if, under BetaEdit, revision chains trail matched distinct facts by ≥10 points on current-value efficacy or stale-value rate, and stale-key removal closes ≥50% of that gap.
- **Kill** if the gap is under 5 points at 3k writes.
- AlphaEdit runs afterwards to report the soft-penalty case; it does not decide GO or kill.

### R2 · More than a cache? (v1 #2)

*Was "Keep the state"; absorbs #7 (as phase 2) and #13 (as the measurement layer) · Tier A · working × latent → factual × latent · ~100 GPU-h*

**What the survey adds.**
- Banking the recurrent state moves it into the survey's "latent repository" sub-area (M+, MemoryLLM, MemoryVLA). It also promotes the state from working memory to factual memory about the scene.
- The survey's paper list has no 3D entries.
- A third-party study note on the survey argues that latent memory may be "just state or caching", not true memory. This card answers that critique directly.
- It also gives a secondary test of the timing claim (#5 leads it): one unchanged latent object becomes long-term memory only through *when* it is written and read.

**Changes from v1.**
- **The headline is the cross-session cache test alone.** Within-session restore after resets becomes secondary, because Scal3R, ReCal3R and Info3R crowd it.
- **The weak controls become a byte- and pass-matched frontier.**
  - On one side, the snapshot: about 1.2 MB if the state is 768×768 in fp16 (confirm in code), with zero extra passes.
  - On the other, JPEG keyframes of equal total bytes re-fed through a cold state. Select them by camera-overlap retrieval (as in the survey-listed Context-as-Memory) and by a ground-truth oracle.
  - **New decision rule (cross-session only, which now decides GO or kill on its own):**
    - **Kill** if cold-state re-feeding of byte-matched, overlap-selected keyframes comes within 2 points of the snapshot. The metric is session-2 pose recall at 25 cm / 10°, averaged over all 7-Scenes and 12-Scenes revisit pairs.
    - **GO** if the snapshot beats that arm by ≥5 points, as in v1.
    - Depth AbsRel is secondary.
    - Within-session restore is reported but does not decide GO or kill.
- **Add a fourth restore operator, "attend, don't replace".** The decoder reads the snapshot without overwriting its state, in the style of Memorizing Transformers and M+. It is out of distribution for a frozen decoder, so a failure is itself a finding.
- **Add a test only latent memory can pass,** for the survey's §7.1 "retrieval to generation" frontier. Query the banked state with CUT3R's raymap at session-2 poses, on views that no single keyframe covers. Apply the same query to a cold state fed byte-matched frames (~3 GPU-h).
- **Drop the bank-budget eviction sweep,** since storage is not the constraint. Replace it with a retrieval-interference curve: false-restore and harm rates against the number of banked snapshots.
- **Report "frame-bound" memory explicitly.** Stratify results by viewpoint distance and by drift between sessions. Add a do-no-harm rate: how often warm start does worse than cold start.
- **Phase 2 (old #7)** compares a kept scene as latent state, text-like tokens or weights at matched bytes, and adds an explicit-map arm. Note that per-scene adapters are not consolidation, because they cannot interfere with each other.

**Pilot.** About 10–12 GPU-h on static 7-Scenes and 12-Scenes. RIO10 comes in only after a GO.

### R3 · Certified retrieval timing (v1 #8)

*Top of Tier B · experiential × token-level · 25–40 GPU-h plus API*

**What the survey adds.**
- None of the 46 experiential × token-level works in the survey's list certifies that memory does no harm compared with no memory.
- The survey's §7.2–7.3 push toward learned memory managers (e.g. Memory-R1, Mem-α) makes the question sharper. Survey-listed MemRL is a learned read-side manager of this kind. Does a learned utility score keep within a harm budget without calibration?

**Changes from v1.**
- **Cut the write gate.** It cost about 80 of the 160 GPU-hours, and ACE, ReMe, Janus and Recuris already crowd it.
- **Make the guarantee a wrapper over any utility score**: a hand-feature score, MemRL Q-values, ExpeL-style counts, or retrieval cosine.
  - Headline result: the same score with and without conformal calibration as the memory drifts.
- **Go API-first,** with a ≤1.5B model as the ablation that supplies log-probabilities. Run the no-memory arm once per item and cache it.
- **Add an abstraction axis**: raw cases versus distilled strategies versus hybrids (ExpeL), at matched tokens. This tests the survey's emphasis on strategy-level memory.
- **Make demotion certified.** An anytime-valid sequential test demotes a harmful insight from the strategy store to the raw-case store. Lineage depth from v1 #15 becomes a feature.
- **Add a reusable leaderboard metric:** "harm-budgeted net gain" over compute-matched no-memory sampling, across at least 4 survey-listed builders.

**Go/kill.**
- *Harm* is the share of items solved by the cached no-memory arm but failed with memory. The *noise floor* is the same quantity between two no-memory resamples.
- **GO** if both hold:
  - late in the stream, harm is ≥5 points above the floor on ≥2 builders, at least one of them not Dynamic Cheatsheet;
  - the conformal wrapper over at least one utility score meets α while keeping ≥50% of the memory gain of the uncalibrated score.
- **Kill** if harm is under 2 points above the floor on every builder.

### R4 · Forgetting, not frequency (v1 #5)

*Tier B · latent → parametric, working → factual · 40–60 GPU-h, mostly CPU*

**What the survey adds.**
- This is the cheapest direct test of the survey's central claim that short- and long-term memory come from the timing of formation, evolution and retrieval.
- It also corrects v1. HOPE's fast level is Titans, which already has a forget gate and momentum. So "HOPE is degenerate" is probably wrong.
- The new headline: **forgetting, not update frequency, is what creates a long-term store.**

**Changes from v1.**
- **The core experiment is a 2×2 on Zoology MQAR with session resets:**
  - fast-level forgetting on or off;
  - update-frequency separation on or off.
  - About 8–10 GPU-h, 3 seeds, and a small model of at most 20M parameters.
  - Pre-registered prediction: the forgetting effect on post-reset recall is at least 3× the frequency effect.
  - If frequency adds ≥25% relative post-reset recall beyond forgetting, report a positive result for frequency. Anything between this and the ≥3× prediction is reported as inconclusive.
- **Add a per-level "timescale certificate"**: recall after a reset, as a fraction of a slow-only learner's recall. It is a form-independent test of whether a level is working or long-term memory. This also addresses the survey filing Titans under latent memory but Lightning Attention under parametric.
- **Add a HOPE attribution appendix** (≤15 GPU-h): clamp the decay, zero the momentum and equalize the update periods.
- **Add explicit-transfer baselines:**
  - M+-style copy-on-forget;
  - session-end distillation, as in the survey-listed SELF-PARAM.
- **Cut** Rotated MNIST, the CUT3R stretch goal, and the "spacing optimum" headline (already shown by Mozer 2009 and Benna & Fusi 2016).

### R5 · Overwrite or wait? A closed-form threshold (v1 #10)

*Tier B · factual × token-level · 30–40 GPU-h plus ~$300–500 API*

**What the survey adds.**
- The survey lists RGMem, which reports a critical update threshold (θ = 3) that it says is the same across datasets.
- A Bayes-optimal overwrite rule predicts that this threshold should move with how often facts change and how noisy observations are. That makes RGMem's claim directly falsifiable.
- The survey's §7.3 also asks whether rules like this should be learned instead of derived.

**Changes from v1.**
- **Lead with theory.** The threshold is k\*(h, ε) ≈ ⌈log((1−h)/h) / log((1−ε)/ε)⌉, where h is the change rate and ε the observation noise. Latest-wins, k-confirm, RGMem, EMA and Memini are special cases or approximations of it.
- **Admit the weak point up front.** When h and ε are constant, the Bayes gate is just a tuned count threshold. The main plot therefore shows the gain over the best single threshold against the spread of k\* across the stream.
- **New kill rule, both conditions required:**
  - a gain of ≥5 points on streams with mixed reliability;
  - a gain of ≤1 point on homogeneous streams, as the theory predicts.
- **Test RGMem's threshold in weeks 1–2** (CPU plus ~$50–100 of API). Sweep θ across change and noise rates and include PersonaMem. If the best θ moves by 2 or more, the invariance claim is falsified.
- **Compare a learned gate with the derived one.** Train a LoRA update/no-update classifier on Qwen2.5-1.5B with the same calibration data, and plot both against calibration size. Prediction: the derived gate ties or wins below about 1k examples.
- **Stop claiming version history** (Zep and timeline memory already do it). Add a flip-cost metric for trustworthy memory: how many contradicting statements it takes to overwrite a slot.

### R6 · Ghosts in the pointer memory (new)

*Tier B · factual × latent · ~40 GPU-h*

**Idea.** In explicit learned 3D memories, such as Point3R pointers and StreamVGGT KV tokens, every new frame reads the stored entries. Entries left by moved or removed objects may therefore corrupt current depth and pose, not just leave debris in the map.

The card first tests whether that happens, by deleting stale entries with ground-truth knowledge on 3RScan rescans. Only if it does, it compares a training-free deletion rule with the attention-, similarity- and importance-based eviction used by published 3D KV methods. The deletion rule uses free-space contradiction: an entry is dropped when new depth shows empty space where it sits (credited to Keller 2013 and ReFusion).

**What the survey adds.** This tests the persistence-of-importance hypothesis from survey-listed Scissorhands, under world change: entries that mattered may stop mattering, and attention-based scores may not notice. It also places "updating latent factual memory" in a cell the survey leaves empty for 3D.

**Gate, ~6 GPU-h plus 1.5 weeks of engineering on a Point3R harness** (StreamVGGT as the second model; CUT3R/TTT3R is measured only, since its state has no per-entry deletion).
- Data: 20 3RScan pairs, each with at least 3 changed objects, plus 5 unchanged pairs.
- Four conditions:
  - carry the memory over;
  - reset;
  - carry with ground-truth removal of stale entries;
  - carry with removal of an equal number of random entries.
- **Primary metric (pre-registered):** depth AbsRel on changed-object pixels. Pixels whose rays pass through vacated space, and pose ATE, are secondary.
- **GO** if all three hold:
  - ground-truth removal improves the primary metric by ≥10% relative over carry, with a 95% bootstrap CI over pairs excluding 0;
  - static-pixel AbsRel changes by ≤2%;
  - random removal of an equal number of entries gives less than half that gain.
- **Kill** below a 5% gain. The topic would then be map hygiene, which existing work already covers.
- **Between 5% and 10%:** extend once to 40 pairs, then apply the same rule.

**Closest work.**
- Map-level change handling: [GaME](https://arxiv.org/abs/2506.06909), [ReFusion](https://arxiv.org/abs/1905.02082), [POCD](https://arxiv.org/abs/2205.01202) and [Changes in Real Time](https://arxiv.org/abs/2511.12370).
- Write-side gating: [RayMap3R](https://arxiv.org/abs/2603.20588).
- Pointer and KV-cache management: [Ray-Aware pointer memory](https://arxiv.org/abs/2605.05749) and [GHOST](https://arxiv.org/abs/2605.15852). GHOST lists dynamic scenes as an open problem.
- "Argos" (2610.10181) appeared in one search snippet and must be verified before it is cited.

### R7 · Keep the index, evict the text (v1 #3)

*Was "Keep the address" · Tier B (was A) · factual, token-level → parametric · 80–100 GPU-h*

**Why it dropped a tier.** Two rivals in the survey's own list threaten it:
- "Self-Updatable LLMs by Integrating Context into Model Parameters" (SELF-PARAM) may make the address unnecessary.
- "Pretraining with hierarchical memories" selects parameter blocks by address and uses no context tokens at all.

Reading the card against the survey also exposed three design flaws:
- atomic facts save at most about 3× in tokens;
- the natural-language stub nearly repeats the question;
- "passes with the full text in context" does not show that a fact is hidden in the weights.

**Changes.**
- **Store per-entity profiles** of about 80–150 tokens, so 10× savings are possible.
- **Make context tokens the headline currency.**
- **Add questions where the context budget actually binds:** cross-episode 2-hop and aggregation over many people.
- **Redefine "hidden":** the fact must still carry at least 50% of the log-probability gain from its write (O'Neill's trace measure).
- **Change the address arms.** The primary addresses become a random entity code and an invented nickname; the stub becomes a control.
- **Add two rivals:**
  - a LoRA bank selected by the store's key;
  - SELF-PARAM-style writes.
  - **Kill at the 0.6B pilot** if SELF-PARAM makes the address unnecessary.
- **Share #1's stale-value rate** for facts that were later updated.

### R8 · Rehearse or advance? (v1 #4)

*Was "Rewrite or advance?" · Tier B · buffer → latent, working memory · ~100 GPU-h*

**What the survey adds.** This card changes only *when* the fixed 3D state is written to (replay fraction ρ, write scale β), at equal compute. It then measures the state's effective memory horizon H with read-only probes. That gives a clean test of the timing claim:
- **Confirmed** if H at 25% replay is at least 10× H with no replay (about 3 frames).
- **Falsified** if H stays within 2× for every setting. A separate store (#2) would then be needed.
- **Between 2× and 10×:** partial support. Continue only if the reset-and-rebuild kill rule below is not triggered.

Either result is publishable, which removes the card's old all-or-nothing risk.

**Changes.**
- **Add a reset-and-rebuild baseline at equal passes,** the 3D version of survey-listed IterResearch. **Kill** if it comes within 3% ATE of interleaved replay.
- **Credit each replay policy to its source:**
  - anchor replay → StreamingLLM's attention sinks;
  - coverage replay → Context-as-Memory;
  - discrepancy replay → MIR.
- **Check co-visibility with geometry** rather than camera overlap alone.
- **Share one within-session table with #2**, comparing three ways to bring the past back: restore a snapshot, reset and rebuild, or interleave rehearsal.
- **Run sweeps on the 224-pixel checkpoint.**

---

## Tier C, merged and dropped

| Idea | Status | One line |
| --- | --- | --- |
| #9 Rent, buy, or let it expire | C | Survey-listed Memory^3's frequency-based memory hierarchy is the uncited static case of rent-or-buy. LMLM argues that changing facts should never enter the weights. What survives is the claim that frequency-only placement misplaces facts that are frequently queried but volatile. It absorbs #14's corroboration gate and a new ~8 GPU-h "consolidation debt" test as its expiry cost. ~75 GPU-h. |
| #11 Folding session states | C | Turning contexts into parameters is an established line in the survey's factual × parametric cell (MAC, SELF-PARAM). v1's storage claim is false at N ≤ 100. It becomes a diagnostic sharing #5's MQAR harness. |
| #6 Displacement-aware experience memory | C | Survey-listed MemRL already handles read-side old-vs-new competition on ALFWorld. What remains is a cheap breakdown of backward-transfer loss by formation, evolution and retrieval. |
| #15 Depth, not time | C | ACE already names and fixes "context collapse", and the survey's tree memories pre-empt the O(log N) fix. Its lineage-depth feature moves to #8. |
| #12 KV eviction audit | C | Now tests Scissorhands' hypothesis by name. The fix becomes head-selective retention in the style of RazorAttention. It stays a diagnostic chapter of Arc 1, cut to ~80 GPU-h. |
| #17 Budgeted 3D episodic memory | C | Survey-listed Embodied VideoAgent is close. It becomes a test of memory structure (flat, graph or hierarchy) against uniform subsampling, which speaks to the survey's complexity-bias critique. |
| Start-state pool (new) | C | A small pool of CUT3R/TTT3R initial states tuned per domain, in the survey's sparsest cell (experiential × latent). The only new question is how long start-state experience survives being overwritten. ~80 GPU-h. |
| RL memory-manager audit (new) | C | An inference-only audit of the released Mem-α checkpoint. Swapping which operation the RL policy picks with what it writes measures where its gain comes from. Partly pre-empted (MINTEval already logs Mem-α's operations). |
| Tiers or timing (new) | C | Replays identical content through tiered agent memories versus timing-only routing. It is the external-store twin of #5. Null results are likely. ~20 GPU-h plus $40. |
| #7, #13 | merged into #2 | #7 becomes #2's phase-2 paper and #13 becomes its measurement layer. |
| #14, consolidation debt | merged into #9 | Both become parts of #9. |
| Generate-vs-retrieve router (new) | merged into #2 | Its headline claims are partly scooped. |
| #16 Replay laws for ViTs | dropped | Replay buffers fall outside the survey's scope, its controller is scooped (2602.22479), and it is the card furthest from the main thesis. |

## Survey claims the portfolio tests

The table lists which cards test each claim. The timing claim is the headline of #5 only. Do not let every card claim to test it; reviewers will read that as inflated framing.

| Survey claim | Where it comes from | Tested by | Prediction |
| --- | --- | --- | --- |
| Short- vs long-term memory come from the timing of formation, evolution and retrieval, not from separate modules | Snippet; verify in the PDF | #5 (lead); #4 and #11 also primary; #2 secondary | Forgetting, not frequency, makes a long-term level (#5). Replay may or may not extend a fixed state's horizon (#4). The same latent object can become long-term memory (#2). |
| Model editing is the updating operator for parametric memory | Snippet | #1 | Protected-set editors only add; updating needs explicit forgetting. |
| Memory management should move from hand-built rules to learned/RL policies (§7.2–7.3) | Table of contents | #8, #10, RL audit | Learned utilities need calibration (#8). A derived gate ties a learned one when data are small (#10). |
| From memory retrieval to memory generation (§7.1) | Table of contents | #2, #3 | Generative read-out from latent state (#2). Cue-driven regeneration from weights (#3). |
| Persistence of importance (survey-listed Scissorhands) | Paper list | Ghosts, #12 | Fails after world change and on long-gap revisits. |
| RGMem's dataset-invariant threshold (survey-listed) | Paper list | #10 | The threshold moves with change rate and noise. |
| Memory^3's frequency-based placement (survey-listed) | Paper list | #9 | Fails for frequently queried facts that change. |
| Latent memory is "just state or caching" | Third-party study note | #2 | The banked state beats byte-matched frames, or it is only a cache. |
| Parametric memory suffers catastrophic forgetting | Third-party summary | #1, #3 | Revised facts show over-retention instead (#1). Forgetting is partly loss of access, not storage (#3). |
| The survey undervalues simple baselines | Third-party critique | Brute-force rows in #1, #2, #8; #17 | Every main table includes the simple baseline. |

## Caveats

- **The survey text itself was not read.** Claims marked "snippet" or "third-party" may not be the survey's own wording.
- **Overlap descriptions rest on search snippets** for MemRL, RGMem, PEAK, WISE, ELDER, M+ and Memory^3. One identifier seen during the search (2609.24238) could not be verified and must not be cited.
- **Survey vocabulary helps when pitching to agent-memory venues, but it adds no novelty.** 3D venues will judge #2, #4 and Ghosts on 3D baselines, not on taxonomy.
- **"Why not keep it in a text store?" is now the obvious reviewer question** for #1, #3 and #9. Each needs an in-context reference row and a stated reason why weights matter.
- **GPU figures are planning estimates.** Confirm the CUT3R raymap path, BetaEdit's GPT2-XL support and the Mem-α checkpoint before budgeting.

Full per-card outputs from this pass are in [`data/pass3_survey_refinement.json`](data/pass3_survey_refinement.json).
