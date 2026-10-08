# Full project plans

These are the detailed plans behind each card in [memory-management-ideas.md](memory-management-ideas.md), in the same rank order. Agents wrote them in two rounds on 2026-10-08. The prior-work notes rest on web-search snippets, because paper pages could not be opened from the environment. Read the papers listed under closest work before committing.

For plans 1, 3, 5, 11 and 16, the second round re-checked prior work after the first plan was written. Read each plan together with its "Second-round re-check" section, which says what must change.

Correction: plan 3's re-check says Harrington et al. (arXiv 2607.07847) could not be found. A separate search confirmed the paper exists: *When Does Continual Learning Require Learning*, Jul 2026.

## 1. Garbage in the Null Space: Superseded Keys Block Fact Revision in History-Aware Sequential Editors

*First-round title: Do Sequential Knowledge Editors Collapse in Writes or in Live Facts? Superseded-Key Protection and Garbage Collection in Null-Space Editors*

**Judge:** overall 7.5/10, tier A. Scores out of 5: novelty 4, low compute 4, impact 3, speed of first signal 5.
This plan has the sharpest falsifiable mechanism in the set, and the recheck strengthened it rather than weakening it. sLKE/MoKE (ACL 2025) document that lifelong editors fail to overwrite, but nobody explains why. The 2026 history-aware null-space editors (BetaEdit, OrthoEdit, EvoEdit) are the current SOTA, and they predict near-total blocking of a revision, not merely 1/(n+1) attenuation, because the revised subject key already lies in the protected span. A CPU closed-form toy (days 1-2) and about 10 GPU-hours on GPT2-XL give a GO/NO-GO within two weeks. The fix (remove the stale key, then downdate or recompute the projector) is training-free and cheap. It is memory management in the literal sense: garbage collection and eviction inside the weights store. It speaks less directly to cross-store consolidation, but it is a mechanism that store-to-weights pipelines (P5, P9) would rely on.

**Biggest weakness:** The phenomenon is already known empirically (sLKE, S2RKE same-subject key cosine of about 1), so the contribution stands or falls on the closed-form diagnosis fitting real editors. AlphaEdit's z-margin or BetaEdit's internals may already avoid the blocking. Read BetaEdit's analysis section and 2606.26783 in full before week 2.

**One line.** In AlphaEdit-family editors, ROME/MEMIT keys depend only on the subject, so every superseded write stays in the protected Gram matrix. Each new revision of a fact then fights all its past values and realises only about 1/(n+1) of its update. A write ledger that exactly downdates superseded keys, with optional compaction, should make collapse scale with live facts rather than total writes.

**Memory store.** weights

**Task.** Lifelong knowledge editing with revision chains, where the same (subject, relation) is overwritten 2-10 times in chronological order. The primary data is AToKe-ME, real YAGO/Wikidata career and marriage chains. CounterFact-derived synthetic chains with a controlled number of revisions are used for the scaling grid, with zsRE/CounterFact distinct-fact controls. The models are small locate-then-edit targets: GPT2-XL and phi-1.5, plus one Llama3-8B confirmation.

**Gap in the big picture.** This targets the weights row of the big-picture document, where shared parameters interfere and protecting them costs plasticity. It is also the weight-store analogue of the lesson that 'useful memories become faulty when continuously updated' (arXiv 2605.12978). Sequential editors collapse after a few thousand edits, and the field explains this with total edit count, null-space saturation (AlphaEdit reproducibility study, 2606.26783, as cited) or weight-norm growth (ENCORE). None of these separates writes from live facts. When a fact is revised, the old write becomes garbage, yet protected-set editors keep defending it. That is exactly the consolidation question of what to keep and what to evict, posed for the weight store.

**Hypothesis.** On chronologically ordered AToKe-ME revision chains, vanilla AlphaEdit on GPT2-XL will score at least 10 points lower on current-value efficacy for revision 3 and later than on write-count-matched distinct-subject facts. The fraction of each revision's residual that is actually realised will follow the closed-form harmonic attenuation predicted from the n superseded copies of the subject key in the protected Gram matrix (R^2 >= 0.8). Exactly downdating those superseded keys will recover at least two-thirds of the gap, while OneEdit/WilKE-style delta rollback alone will recover less than one-third.

**Method.**

Diagnosis. In AlphaEdit, MEMIT_seq and NSE (official AlphaEdit repo), each layer's update solves [P(K K^T + C_p) + lambda I] Delta^T = P K R^T. Here P is the null-space projector of preserved Wikipedia keys and C_p is a running Gram matrix: cache_c += k k^T after every batch, kept as a sum with no per-fact identity. The key k is the subject's last-token MLP input averaged over random prefixes, so it is relation-independent. After n earlier writes to the same subject, C_p therefore holds about n copies of k k^T, and the new edit realises roughly (k^T P k)/((n+1) k^T P k + lambda), or about 1/(n+1), of the per-layer residual. Compounded over MEMIT's residual spreading across 5 layers, the predicted realised fraction for revisions 1 through 5 is about 1.0, 0.75, 0.60, 0.48 and 0.39. The superseded value also stays protected, leaving a ghost. Fix (write ledger). For every write, store (subject id, relation id, object, batch, per-layer key k_l at write time, the per-edit Delta contribution). On supersession of (s, r), (i) exactly downdate C_p,l -= k_l k_l^T for the stale entry, which is valid because C_p is additive; (ii) write the new value, which overrides the old one because the residual is computed against the current output; (iii) re-add the new key so that the new value becomes the protected one. Optionally (iv) compact: once the garbage ratio exceeds tau, reset to W0 and batch-edit all live facts in one AlphaEdit solve. Ablations: a 2x2 factorial of {delta rollback no/yes} x {key downdate no/yes}; downdate with the stored key vs a key recomputed under the current model; turning protection off entirely (C_p = 0); and tau in {0.25, 0.5, 0.75}. A negative control edits the same subject with different, still-live relations: their keys are identical too, so garbage collection should not help, which separates garbage from genuine conflict. A CPU-only linear associative-memory simulation (d = 256, random keys with shared subject keys, AlphaEdit's closed form) validates the attenuation law and the garbage-collection fix before any GPU run.

**Datasets.**

- AToKe-ME (github.com/Arvid-pku/ATOKE, verified): 4,609 real same-(s,r) chains with >=2 revisions (14,434 writes), 2,603 chains with >=3 (10,422 writes), 692 chains with >=5 (4,008 writes), and 73 A->B->A reversion chains; mostly playsFor, isMarriedTo and isAffiliatedTo; includes paraphrase, question and time-qualified prompts
- AToKe-SE (same repo, 8,819 single edits): relation-matched distinct-fact control
- CounterFact (AlphaEdit/ROME dsets): synthetic revision chains with R in {1, 2, 3, 5} built by sampling successive targets from the same relation's object pool; neighborhood prompts for specificity
- zsRE (AlphaEdit dsets): distinct-fact QA control
- GLUE subset (SST-2, MRPC, CoLA, RTE, NLI, MMLU) via AlphaEdit's glue_eval for general ability
- Optional: WikiBigEdit (github.com/ExplainableML/WikiBigEdit, verified) for real Wikidata updates across snapshots, filtered to (s,r) pairs changed in >=2 snapshots

**Models / checkpoints.**

- openai-community/gpt2-xl (1.5B; official AlphaEdit hparams, layers 13-17)
- microsoft/phi-1_5 (1.3B; official AlphaEdit hparams in repo)
- meta-llama/Meta-Llama-3-8B-Instruct (official AlphaEdit Llama3-8B hparams; one confirmation point only, on an 80GB card)
- Optional: Qwen/Qwen2.5-1.5B-Instruct (needs a causal-tracing layer search and new covariance stats)

**Baselines.**

- AlphaEdit vanilla (github.com/jianghoucheng/AlphaEdit, ICLR 2025)
- MEMIT_seq and NSE from the same repo (both carry cache_c protection)
- MEMIT without an edited-key cache (protection off; isolates the C_p term)
- AlphaEdit with C_p = 0 (shows that garbage collection is not just turning off protection: distinct-fact retention should drop)
- Rollback-only, re-implementing OneEdit/WilKE: subtract the superseded edit's stored Delta and keep C_p
- ENCORE-style norm-constrained editing, or a reimplemented norm penalty, plus a norm-matched control for the competing weight-norm-growth explanation
- UltraEdit (github.com/XiaojieGu/UltraEdit, TMLR'26; 'subject-free' lifelong editor)
- GRACE and WISE via EasyEdit (side-memory editors whose codebook replaces the value on key collision)
- Sequential LoRA / FT-L
- Oracle: batch AlphaEdit of the current live-fact set from base weights (the compaction upper bound)

**Metrics.**

- Current-value efficacy: P(o_current) > P(o) for every superseded and original object in the chain, plus paraphrase and question generalization
- Ghost rate: fraction where a superseded object (not the original) is the argmax among chain objects
- Realised residual fraction per revision at the z-layer, ||z_after - z_before|| / ||z* - z_before||, against the analytic 1/(n+1) prediction (R^2)
- Specificity: CounterFact neighborhood score and retention of earlier distinct edits
- General ability: GLUE subset and MMLU via glue_eval
- ||W - W0||_F per layer, and the effective rank and condition number of C_p against writes and against live facts
- Scaling fit: degradation regressed on log(live facts) vs log(total writes); report which explains more variance, with and without garbage collection
- Secondary: AToKe time-qualified (historical) recall, to show garbage collection does no worse than vanilla on history

**First experiment (go/no-go).**

Days 1-2 (CPU, no GPU): linear associative-memory simulation using AlphaEdit's exact closed form, to confirm harmonic attenuation for revision chains and show that the rank-1 downdate removes it. Days 3-10 (one A100, about 8-10 GPU-hours): run the official AlphaEdit code on GPT2-XL with the ROME/AlphaEdit Wikipedia covariance stats. Data: 1,000 AToKe-ME chains of length >=3, truncated to 3 revisions (3,000 writes), in chronological order, batch size 100, with each chain's revisions forced into different batches. Compare against 3,000 distinct-subject AToKe-SE facts (matched writes) and against the 1,000 final values edited once (matched live facts). Conditions: AlphaEdit vanilla, +downdate, +rollback and +both; MEMIT_seq vanilla and +downdate; 6 runs of about 1-1.5 GPU-hours each. Log realised residual fraction per revision, current-value efficacy, ghost rate and ||W - W0||. GO if all three hold: (a) vanilla AlphaEdit trails the distinct-fact control by >= 10 points on revision 2 and later in current-value efficacy or ghost rate; (b) realised fraction tracks the harmonic prediction; (c) downdate closes >= 50% of the gap while rollback-only closes clearly less.

**Kill criterion.**

Drop or pivot if any of these hold. (1) At 3k writes on GPT2-XL, vanilla AlphaEdit shows less than a 5-point gap between revision chains and matched distinct facts in both current-value efficacy and ghost rate; the z-margin may absorb the attenuation. (2) The realised residual fraction does not fall with the number of prior same-subject writes, which would mean the mechanism is wrong. (3) Rollback-only fixes the gap as well as the downdate does; then OneEdit/WilKE already cover it and there is no delta. (4) A re-run novelty search, or a full read of 2606.26783, shows that superseded or duplicate keys in AlphaEdit's cache have already been diagnosed and fixed.

**Compute.** 150 GPU-hours over 10 weeks. 1x A100/H100 80GB (GPT2-XL and phi-1.5 also fit on a single 24-48GB card in fp32; the 80GB card is needed only for the Llama3-8B fp32 confirmation). The toy simulation and analysis run on CPU. Ledger storage is about 1.3 GB per 10k edits for GPT2-XL (6400-dim keys x 5 layers), kept on CPU or disk.

**Risks.**

(1) Novelty could not be swept, because of the exhausted search budget and arXiv DNS failure. Concurrent 2026 work, or 2606.26783 itself, may already diagnose duplicate keys. (2) The z-optimisation margin (clamp_norm_factor, 20 gradient steps) may absorb 25-40% attenuation, so binary efficacy hardly moves. In that case the result lives in probability margins and longer chains, which makes a weaker paper. (3) The fix is engineering-trivial (a rank-1 downdate), so the contribution must rest on the closed-form diagnosis, the factorial comparison with rollback and the live-facts scaling law; reviewers may call it narrow to the AlphaEdit family, although NSE, MEMIT_seq and the many AlphaEdit derivatives share the cache. (4) Keys drift as earlier layers are edited, so downdating with the stored k_old is inexact; the plan ablates this against recomputed keys. (5) Same-subject edits on different relations that are both live share the key and are not fixed by garbage collection; this is scoped as a negative control, not solved. (6) AToKe is about 83% playsFor with obscure entities, so results may be domain-specific; the CounterFact synthetic chains broaden coverage. (7) 'Garbage' means superseded for current-time queries only; garbage collection discards history, which AToKe's time-qualified prompts measure. (8) Editors without subject keys (UltraEdit, GRACE) may already be immune, which narrows the claim but is itself a finding.

**Target venue.** ICML 2027 (late-January deadline) or ACL/EMNLP 2027 via ARR; a diagnostic short paper also fits the KnowledgeNLP or Model-Editing workshops

**First-round prior-work verdict: partially_novel.** Several pieces already exist: re-editing the same fact with rollback (OneEdit's coverage conflict, WilKE), benchmarks of facts revised over time (ATOKE), same-subject interference (2502.06868) and norm-growth explanations of collapse (ENCORE). The defensible delta has three parts: (i) a closed-form diagnosis that protected-Gram editors (AlphaEdit, MEMIT_seq, NSE) count every superseded write of a subject, because the keys are relation-independent, giving harmonic attenuation of revisions plus ghost values; (ii) an exact rank-1 downdate ledger with compaction, shown factorially to beat rollback alone; (iii) evidence that collapse scales with live facts rather than total writes. Confidence is low. The web search budget for this turn was exhausted and arXiv was unreachable via DNS, so the only verification was the code and data on GitHub (AlphaEdit's additive cache_c, OneEdit's rollback, AToKe-ME chain statistics). Re-run the novelty search, and read 2606.26783 in full, before committing.

**Closest work found in the first round:**

- [OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System](https://arxiv.org/abs/2409.07497) — arXiv 2409.07497 (2024). This is the most direct threat. It explicitly handles 'coverage conflict', where the same (subject, relation) is edited again, by rolling back the cached parameter delta of the earlier edit before applying the new one. That covers part (ii) of the idea, cancelling the old edit's delta. As far as I know it does not remove stale keys from a null-space protected set, and it does not study collapse scaling over 2-10-revision chains. The page could not be fetched (DNS failure), so this description comes from the idea text and prior knowledge.
- [WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing](https://arxiv.org/abs/2402.10987) — arXiv 2402.10987 (2024). The idea text says it proposes rollback editing for lifelong editing. If so, the 'undo the old delta' operation already exists, and the delta has to be the null-space/protected-set bookkeeping. Not fetched; unverified in this run.
- [History Matters: Temporal Knowledge Editing in Large Language Model (ATOKE benchmark, METO)](https://arxiv.org/abs/2312.05497) — AAAI 2024 (arXiv 2312.05497). From prior knowledge, not verified in this run: it builds a benchmark of facts updated several times over time and shows that editors overwrite or confuse historical and current values. That partly pre-empts the 'revision chains' task framing and the ghost or stale-object metric. It does not use null-space editors or protected-set garbage collection.
- [Unveiling the Pitfalls of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2310.02129) — ICLR 2024 (arXiv 2310.02129). From prior knowledge, not verified in this run: it introduces knowledge-conflict settings in which sequential edits to related or identical facts interfere, and shows that editors leave inconsistent residues. This overlaps with the claim that stale edits leave 'ghost' associations, but it offers no fix based on a ledger or null-space.
- [Related Knowledge Perturbation Matters: Rethinking Multiple Pieces of Knowledge Editing in Same-Subject](https://arxiv.org/abs/2502.06868) — arXiv 2502.06868 (2025). Studies interference between edits that share a subject. Close in spirit to same-(s,r) supersession, but as cited in the idea it covers different relations of the same subject, not repeated revisions of one relation. Not fetched.
- [AlphaEdit reproducibility study (as cited in the idea)](https://arxiv.org/abs/2606.26783) — arXiv 2606.26783 (2026). Per the idea text, it suggests null-space saturation as the cause of divergence after a few thousand edits but does not confirm it. If it also tests duplicate or revised keys in K_p, the core diagnosis would be pre-empted. Not fetched, so its content could not be checked.
- [Lifelong Knowledge Editing requires Better Regularization (ENCORE)](https://arxiv.org/abs/2502.01636) — arXiv 2502.01636 (2025). From prior knowledge, not verified in this run: it attributes sequential-editing collapse to growth in weight norm and overfitting, and fixes it with norm-constrained regularization over 10k edits. That is a competing explanation the idea must beat with its weight-norm-growth metric. It does not address supersession.
- [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355) — ICLR 2025 (arXiv 2410.02355). This is the base method. Its sequential objective accumulates previously edited keys K_p, and the official code keeps a running cache of them. The idea would be a modification of that bookkeeping. Removing a column from the cache is trivial engineering, so the contribution has to be empirical and diagnostic.

### Second-round re-check (live search)

**Revised one line.** Protected-set editors (AlphaEdit, BetaEdit, OrthoEdit, EvoEdit, MEMIT_seq) keep defending every superseded write of a subject. Because subject keys are relation-independent, each revision is harmonically attenuated by soft Gram penalties and almost fully blocked by hard null-space projectors, and a ghost of the old value stays protected. A write ledger that downdates stale keys (or recomputes the projector) and compacts periodically should make degradation track live facts rather than total writes.

**Verdict: partially_novel; recommendation: pursue_with_pivot.**

**What is taken and what is still new.** Twelve searches turned up no work that (a) identifies superseded writes of a subject as garbage left in the protected set of AlphaEdit-family editors, (b) derives the resulting attenuation of revisions and the 'ghost' values they leave, (c) fixes this with an exact downdate ledger and compaction, or (d) shows that collapse scales with live facts rather than total writes. One search aimed squarely at closed-form analysis of duplicate keys versus unique facts found nothing. Several pieces around the idea are taken, though. The repeated-overwrite task and its failure ('cannot precisely overwrite outdated knowledge') come from the ACL 2025 sLKE/MoKE paper. Same-subject key cosine near 1 in MEMIT is reported by S2RKE (2502.06868). The covariance constraint has already been blamed for same-subject failures ('Covariance Trap', 2603.15518). History-aware protected-key null spaces have become the main line of sequential editing: BetaEdit (IJCAI 2026), OrthoEdit (TACL 2026), EvoEdit (2025). Temporal current-versus-history editing is covered by PRISM Edit (2607.11327). Rollback exists in OneEdit (system level) and in training-free edit reversal (2505.20819, ICLR 2026). The 2026 null-space wave widens the defensible delta. Those editors project every new update away from the span of all past edited keys, so a revised subject's new key, which is nearly identical to its old one, should be almost entirely blocked, not merely attenuated. The diagnosis therefore covers the whole family as the field's current best practice, and the fix is to remove the stale key from the protected span and recompute the projector, which no one appears to do.

**Required revisions.** 1) Drop the WilKE rollback attribution. In the search results WilKE does layer selection, not rollback. Cite OneEdit's KG-level rollback and 'Tracing and Reversing Edits' (2505.20819) as the rollback and reversal prior work, and use the latter as an extra rollback-arm baseline. 2) Widen the diagnosis from AlphaEdit's soft cache_c to the 2025-26 history-aware null-space editors: BetaEdit (2605.09285, code at lbq8942/BetaEdit), OrthoEdit (TACL 2026) and EvoEdit (2510.13851). For hard projectors, predict near-total blocking of revisions (realised fraction of roughly 1 - ||P_hist k_new||/||k_new||), not 1/(n+1). The fix there is to remove the stale key's direction and recompute or downdate the eigendecomposition. Validate both regimes in the CPU toy. This is the strongest selling point, because these are the current SOTA. 3) Do not claim the same-subject key-collision observation as new. Credit S2RKE (2502.06868), which reports MEMIT same-subject key cosine of about 1, and 'Beyond the Covariance Trap' (2603.15518). The new parts are supersession, garbage in the protected set, and the live-facts scaling law. 4) The task and benchmark framing is partly taken. Add the ACL 2025 sLKE benchmark (answers to the same concept updated repeatedly) next to AToKe-ME, and add MoKE as a baseline. Position the work as a mechanistic explanation of why parametric editors fail sLKE, with a training-free fix. 5) Add PRISM Edit (2607.11327) and its TimeConflict/TIMECF benchmark for the history-preservation side. Garbage collection discards history, so report historical recall honestly against PRISM Edit, or scope it out explicitly. 6) The kill criterion (4) check of 2606.26783 found nothing in the snippets. That paper reports degradation after about 5k edits and bounded protection, with no duplicate-key diagnosis. Still skim the full PDF and BetaEdit's analysis section before committing, since neither could be opened here. 7) Compute stays within budget. Adding BetaEdit, OrthoEdit and EvoEdit runs on GPT2-XL and phi-1.5 adds about 20-30 GPU-hours. Consider dropping the Llama3-8B point if the budget is tight.

**Baselines to add.**

- BetaEdit (IJCAI 2026, arXiv 2605.09285; github lbq8942/BetaEdit), the history-aware null-space editor, vanilla and with stale-key removal
- OrthoEdit (TACL 2026), the recursive shared null space over previously edited keys
- EvoEdit (arXiv 2510.13851), sequential null-space alignment
- MoKE / serial lifelong editing (ACL 2025), plus its sLKE benchmark as an evaluation set
- Training-free edit reversal from 'Tracing and Reversing Edits in LLMs' (ICLR 2026, 2505.20819) as an alternative rollback arm
- PRISM Edit (2607.11327) as the time-aware comparator on historical versus current recall
- RoSE (2603.15518) as a same-subject comparator, batch setting only

**Closest work (second round):**

- [BetaEdit: Null-Space Constrained Sequential Model Editing](https://arxiv.org/pdf/2605.09285) — IJCAI 2026 (arXiv 2605.09285). Biggest new threat, and also the strongest new motivation. It analyses why 'history-aware' updates keep null-space editors working over long horizons, and builds C^(t) from the original keys plus the keys of all earlier edits, projecting onto its null space. It is the hard-projector version of AlphaEdit's cache_c. The snippets show nothing about superseded or duplicate keys. Under BetaEdit a revised subject's key may sit almost entirely inside the protected span, so revisions could be blocked outright rather than harmonically attenuated. It must be a baseline and a target of the diagnosis.
- [OrthoEdit (TACL 2026)](https://aclanthology.org/2026.tacl-1.51.pdf) — TACL 2026. Recursively builds a shared null space that removes the directions of all previously edited keys, which is the same protected-set bookkeeping. The plan's supersession issue applies here in hard-projection form. Snippets do not mention revisions or garbage keys.
- [EvoEdit: sequential null-space alignment](https://arxiv.org/pdf/2510.13851v2) — arXiv Oct 2025 (2510.13851). Projects edits into the null space of both the original knowledge and previously modified knowledge, so it is another protected-set editor exposed to stale-key protection. No supersession handling appears in the snippets.
- [Serial Lifelong Editing via Mixture of Knowledge Experts (sLKE benchmark)](https://aclanthology.org/2025.acl-long.1492) — ACL 2025 (long 1492). Pre-empts the task framing. It states that lifelong editors fail to precisely overwrite outdated knowledge with the latest value, builds the sLKE benchmark where answers to the same concept are updated repeatedly, and proposes MoKE routing so that each update overwrites the old one. It gives no closed-form diagnosis of protected Gram or null-space keys, no downdate, and no live-facts scaling result.
- [Related Knowledge Perturbation Matters: Rethinking Multiple Pieces of Knowledge Editing in Same-Subject (S2RKE)](https://arxiv.org/html/2502.06868v1) — NAACL 2025 short (arXiv 2502.06868). Already reports that for MEMIT, same-subject keys have cosine similarity close to 1, and that the first edit's efficacy drops when a later edit shares its subject. The 'relation-independent subject key' observation is therefore not new. It covers different relations of one subject, not superseded revisions of one (s,r), and it neither treats protected-set editors nor offers a fix.
- [Beyond the Covariance Trap: Unlocking Generalization in Same-Subject Knowledge Editing (RoSE)](https://arxiv.org/abs/2603.15518) — arXiv 2603.15518 (Mar 2026). Blames the covariance constraint for same-subject editing failures, specifically weaker instruction and paraphrase generalization, and fixes this with isotropic alignment and hierarchical integration. It is batch-only and the authors call it unsuitable for sequential lifelong editing. It shares the 'the covariance term is the culprit' angle but not supersession or garbage collection.
- [PRISM Edit: One Vector for All Temporal Answers](https://arxiv.org/abs/2607.11327) — arXiv 2607.11327 (Jul 2026). Temporal editing where the new answer becomes current but the old one stays valid in historical contexts. It argues the edit site cannot discriminate time and optimises a single shared representation. It adds the TimeConflict/TIMECF benchmark (22,708 records over 24 relations). It competes with the plan on revision-chain and historical-recall evaluation and is the natural foil for 'GC discards history'.
- [Reproducibility Study of AlphaEdit](https://arxiv.org/html/2606.26783v2) — arXiv 2606.26783 (Jun 2026). Confirmed real (Ananth K S and Arya Hariharan). AlphaEdit is stable to 3,000 edits and degrades after about 5,000, protection is 'bounded rather than unconditional', the advantage does not carry over to newer model families, and a fluency-metric bug is reported. The snippets show no diagnosis of duplicate or superseded keys.
- [OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System](https://arxiv.org/abs/2409.07497) — arXiv 2409.07497 (2024). Confirmed. A controller uses a knowledge graph plus rollbacks to resolve knowledge conflicts, a system-level version of rollback-then-re-edit. It does not touch the protected-key cache.
- [Tracing and Reversing Edits in LLMs](https://arxiv.org/abs/2505.20819) — ICLR 2026 (arXiv 2505.20819). Reverses up to 94% of edits training-free, without the edit record. It is an alternative to ledger-based rollback of superseded deltas and a possible baseline for the rollback arm.
- [Lifelong Knowledge Editing requires Better Regularization (ENCORE)](https://arxiv.org/pdf/2502.01636) — arXiv 2502.01636 (2025). The competing explanation, weight-norm growth. The plan already includes a norm-matched control.
- [WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing](https://arxiv.org/abs/2402.10987) — Findings ACL 2024 (arXiv 2402.10987). Confirmed real, but the snippets describe layer selection by pattern matching against 'toxicity buildup and flash'. No rollback mechanism appears, so the plan's 'OneEdit/WilKE-style delta rollback' attribution looks wrong for WilKE.

**Queries run:** AlphaEdit sequential editing same fact edited multiple times stale keys null-space projection conflict (extended); arXiv 2606.26783 AlphaEdit reproducibility; knowledge editing re-editing same subject relation multiple updates overwrite lifelong editing 2025 2026 arXiv; "Beyond the Covariance Trap" same-subject knowledge editing RoSE; "Serial Lifelong Editing via Mixture of Knowledge Experts" outdated knowledge replaced sLKE; BetaEdit null-space constrained sequential model editing 2605.09285; PRISM Edit "One Vector for All Temporal Answers" knowledge editing time; knowledge editing undo or retract previous edit sequential editing reversal rollback locate-then-edit superseded edit 2026 (extended); OneEdit neural-symbolic knowledge editing coverage conflict rollback 2409.07497; WilKE 2402.10987; "history-aware" sequential knowledge editing null space previously edited keys cache covariance 2025 2026; knowledge editing benchmark facts updated repeatedly chain of updates same question multiple edits MEMIT AlphaEdit fail latest answer 2026; sequential model editing collapse number of edits vs unique facts duplicate edits same key interference closed-form analysis linear associative memory

---

## 2. Keep the State: Banking, Recovering and Reusing Recurrent 3D Memory Across Resets and Sessions in CUT3R/TTT3R

**Judge:** overall 7/10, tier A. Scores out of 5: novelty 3, low compute 4, impact 4, speed of first signal 4.
This is the doc's 3D section turned into a question with a clean yes/no oracle test: does a stored recurrent state carry information beyond its source frames (state restore vs re-feeding the same keyframes vs MASt3R+PnP, with identical retrievals)? It is training-free, the pilot costs 15-20 GPU-hours, and it covers the fast state to external store direction directly. An independent search in the m-3d-adapter check found no work that saves and reloads CUT3R/TTT3R/VGGT-family state across sessions, which supports the cross-session claim. It is also the base for the whole 3D thesis arc.

**Biggest weakness:** The state may be little more than a cache of its source frames, or may be bound to session 1's gauge in ways that make read-only decoding brittle. Scal3R, loop-closure systems and the roughly monthly stream of CUT3R follow-ups make a concurrent cross-session paper likely, so speed matters.

**One line.** Treat the recurrent state of CUT3R/TTT3R as memory worth keeping. Bank verified state snapshots on the CPU, keyed by place. Restore them after resets and on revisits, and warm-start a later session from them, all without training. Test whether the stored state does better than re-feeding the stored images to the model, or than loop closure through a pose graph.

**Memory store.** Cross-store: fast state (the CUT3R/TTT3R recurrent state, i.e. state tokens plus the pose-retriever memory) moving to and from an external, place-indexed CPU snapshot bank that persists across resets and sessions

**Task.** Streaming RGB-only 3D reconstruction and pose estimation in two settings. (1) Cross-session (headline): a second session in the same scene is relocalized and mapped directly in the first session's coordinate frame. Data: 7-Scenes and 12-Scenes sequence pairs, plus RIO10 rescans that contain real changes. (2) Within-session recovery: long sequences with revisits, where TTT3R resets or overwrites discard the state. Data: TUM RGB-D, ScanNet, KITTI loops.

**Gap in the big picture.** This targets the doc's fast-state failure ("fixed capacity; lost when the session ends unless consolidated") and its open problem ("consolidation: what should move from context or an external store into the weights, and when"), instantiated in the doc's own streaming-3D section. TTT3R degrades on long sequences and falls back to periodic hard resets that throw the state away, and FILT3R reports scale jumps at those resets. All the 2026 fixes (TTSA3R, MeMix, FILT3R, AFG, ReCal3R, Info3R's Dynamic State Reset) change how the state is overwritten or when it is reset. None of them recovers a state that has already been overwritten, and none carries the state into a later session. Scal3R resets the CUT3R state every 10 frames and closes loops with archived pose tokens plus a GTSAM graph, so it treats the state as disposable. The project also applies two lessons from the doc. First, "storage was never the real constraint": a snapshot is about 1-2 MB, so the cost lies in verification and retrieval. Second, "external memory moves the dilemma to memory access": with a bank, retrieval and verification become the bottleneck, and the method is built around them. Nobody has yet asked, under controlled conditions, whether a learned recurrent 3D state is memory worth keeping and transferring, rather than just a cache of its source frames.

**Hypothesis.** With identical place retrievals, restoring a verified CUT3R/TTT3R state snapshot does better than both alternatives: re-processing the same stored keyframe images through a cold state, and pose-graph-only loop closure. Two targets: (a) in a later session, it relocalizes at least 10 points more 7-Scenes/12-Scenes query frames within 25 cm / 10 deg, with no Sim(3) alignment step; (b) on revisit-heavy sequences, it reduces ATE by at least 20% relative to TTT3R with periodic reset and relative to Scal3R. On no-revisit control sequences it should match TTT3R.

**Method.**

1. Write. Hook the public CUT3R/TTT3R code so that it copies the full recurrent state to a CPU bank: the 768 state tokens plus the pose-retriever memory, about 1-2 MB in fp16. A copy is taken at keyframes (pose change over 0.3 m / 15 deg, or a SALAD novelty gate) and always immediately before a TTT3R reset. Each entry stores a DINOv2-SALAD key, the snapshot's frame-to-world Sim(3), a timestamp and a health score (mean decoder confidence times the TTT3R query-key alignment). Under a fixed budget of B in {8, 16, 32, 64}, eviction maximizes facility-location coverage over the keys, minus penalties for staleness and low health. This is compared against FIFO, random and confidence-only eviction.

2. Retrieve and verify. Each frame queries the bank with FAISS top-k, restricted to entries more than G frames old or from a previous session. A candidate is verified by decoding the current frame against the snapshot in read-only mode (the state is not updated). It is accepted only if decoder confidence exceeds tau, its depth agrees with the live decode (AbsRel < 0.15), and 3 consecutive frames give consistent poses. This rejects false loops from perceptual aliasing.

3. Restore, keeping frames consistent. Anchor3R found that caching frame-dependent pose tokens across coordinate frames hurts, so three operators are compared:
- R1, pose-only: the verified pose in the snapshot's frame becomes a Sim(3) loop factor in a small GTSAM graph on the CPU.
- R2, hard switch: replace the live state with the snapshot and inherit its frame transform, i.e. re-anchor to the older, less-drifted latent map.
- R3, per-token blend: S <- (1-a)*S_t + a*S_j, applied to the image-conditioned state tokens only. The weight a is the TTT3R-style alignment of the current view with S_j relative to S_t.
Further ablations: restore the state tokens only, the pose memory only, or both.

4. Cross-session (headline). Keep the bank after session 1. Session 2 starts from the retrieved, verified snapshot instead of CUT3R's learned initial state, so its poses come out directly in session 1's frame. A change-gated release handles changed scenes: state tokens whose alignment with incoming observations stays low over a window either have their TTT3R learning rate set to 1 or are reset to the initial tokens. This targets stale geometry in RIO10.

5. Controls. The same retrievals are fed to four alternatives, so that any gain from R2/R3 can only come from the stored state:
- pose graph only (the 2608.27529 recipe);
- Scal3R-style re-injection of archived tokens;
- re-feeding the stored keyframe images through a cold state;
- write-protecting the state on revisits (update=False).

**Datasets.**

- 7-Scenes (Microsoft RGB-D 7-Scenes): every sequence in a scene shares one coordinate frame, so it gives cross-session pairs (train seq to test seq). Also used for Acc/Comp/NC and within-session revisits
- 12-Scenes (Stanford): train and test sequences share one frame per room; used for cross-session pairs
- RIO10 (derived from 3RScan, 10 scenes): the _01 train and _02 validation rescans have public GT poses in a shared frame. They contain real object and layout changes, and stats.txt gives per-frame semantic-change, geometric-change and pose-novelty scores. Data is available on request; fallback is 3RScan rescans with their provided rescan-to-reference transforms
- TUM RGB-D: fr3/long_office_household, fr2/desk and fr1/room (loops)
- ScanNet v2: about 15 scenes with clear loops and at least 1000 frames
- KITTI Odometry: 00, 02, 05, 06, 07, 08 and 09 have loops; 01, 03 and 04 are no-loop controls
- NRGBD: reconstruction accuracy at 500-1000 frames, following the TTT3R evaluation protocol

**Models / checkpoints.**

- CUT3R final checkpoint cut3r_512_dpt_4_64.pth (github.com/CUT3R/CUT3R, Google Drive link in the README)
- TTT3R update rule and --reset_interval (github.com/Inception3D/TTT3R); uses the same cut3r_512_dpt_4_64.pth checkpoint
- Scal3R released checkpoint (github.com/NVlabs/scal3r): frozen CUT3R backbone plus about 1% relative-pose-query parameters, with DINOv2-SALAD + FAISS retrieval and GTSAM loop closure
- DINOv2-SALAD place descriptor (github.com/serizba/salad, dino_salad.ckpt; DINOv2 ViT-B/14 backbone facebook/dinov2-base)
- MASt3R naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric (cold-start pairwise relocalization baseline, then PnP)
- hloc SuperPoint + LightGlue (classical cross-session relocalization baseline)
- Optional streaming baselines on a subset: StreamVGGT, Point3R

**Baselines.**

- CUT3R (vanilla recurrent update)
- TTT3R (no reset)
- TTT3R with --reset_interval (as released): the discard baseline
- Overwrite and reset control rules: ReCal3R, TTSA3R, Info3R (Dynamic State Reset), MeMix/FILT3R where code is released
- Scal3R (same CUT3R backbone; archived pose tokens + SALAD/FAISS + GTSAM; resets the state every 10 frames)
- Retrieval + pose graph only on our identical SALAD retrievals (the 2608.27529 / VGGT-Long recipe)
- Image re-processing control: the same retrieved keyframe images re-fed through a cold CUT3R state (HorizonStream-style)
- Write-protect on revisit (update=False), zero cost
- Cross-session: cold start + oracle Sim(3) alignment (upper bound)
- Cross-session: cold start + MASt3R-PnP or hloc relocalization against session-1 keyframes
- Cross-session: cold start with the retrieved session-1 keyframe images prepended (stored images instead of stored state)
- Optional on a subset: StreamVGGT, Point3R

**Metrics.**

- ATE RMSE (Sim(3)-aligned within a session; for cross-session also unaligned in the session-1 frame)
- RPE translation and rotation
- Relocalization recall at (5 cm, 5 deg) and (25 cm, 10 deg); median translation and rotation error
- DCRE on RIO10 (official metric), stratified by the per-frame geometric-change score
- Acc / Comp / NC on 7-Scenes and NRGBD
- Depth AbsRel and delta<1.25 on revisit frames and on the first 100 frames of session 2
- Recovered fraction of reset loss = (ATE_reset - ATE_method) / (ATE_reset - ATE_best_no_reset)
- Loop verification precision and recall; false-loop rate on KITTI and aliased rooms
- FPS overhead, GPU peak memory, CPU bank size in MB
- Gain retained as a function of bank budget B (coverage vs FIFO eviction)

**First experiment (go/no-go).**

This is a two-week, single-GPU oracle-retrieval test that decides whether the stored state carries any information beyond its source images. Budget: about 15-20 GPU-hours.

Days 1-3: Fork TTT3R. Add hooks that dump the state tokens and pose memory to the CPU in fp16, a read-only decode (no state update), and frame bookkeeping. Precompute SALAD descriptors.

Days 4-7, cross-session: Use 7-Scenes (all 7 scenes, 2-3 sequence pairs each) plus 2 rooms from 12-Scenes. Run session 1 and take a snapshot every 25 frames. For every 10th session-2 query frame, pick the oracle snapshot (highest GT frustum overlap) and compare:
- (i) read-only decode against that snapshot, giving the pose directly in session 1's frame;
- (ii) a cold state fed the snapshot's keyframe image, then the query;
- (iii) a cold state fed the last 8 session-1 keyframes, then the query;
- (iv) MASt3R pairwise + PnP.
Also warm-start the whole of session 2 from the best snapshot, and compare it with a cold start plus oracle Sim(3): ATE and early-frame depth AbsRel.

Days 8-12, within-session: Run TTT3R with reset_interval=100 on TUM fr3/long_office_household, TUM fr2/desk and 3 ScanNet loop scenes, using oracle revisit detection. Compare: reset as-is, R1 pose-only + GTSAM, R2 hard switch to the pre-reset snapshot, R3 per-token blend, and write-protect.

GO if both hold: (a) (i) beats both (ii) and (iii) by at least 5 points of recall at 25 cm / 10 deg, or warm-start ATE without alignment is within 1.5x of cold start + oracle Sim(3); and (b) R2 or R3 improves on R1 by at least 10% ATE, or by at least 10% revisit-frame depth AbsRel.

**Kill criterion.**

Even with oracle retrieval, a stored state snapshot may fail to relocalize better than re-feeding the snapshot's own keyframe images through a cold state ((i) <= (ii) within 2 points of recall at 25 cm / 10 deg), and state restore (R2/R3) may give less than 10% ATE gain over pose-graph-only (R1). If both happen, the state is not a reusable memory beyond its source frames: drop the project, or at most write it up as a short negative-result workshop note.

If only the cross-session test fails, drop the headline and pivot to a within-session recovery paper benchmarked against Scal3R. If only the within-session test fails, keep cross-session warm start as the sole contribution.

**Compute.** 120 GPU-hours over 12 weeks. One 24-48 GB GPU (RTX 4090, A6000 or L40S) and 64 GB of CPU RAM. Everything is inference only. CUT3R/TTT3R run at about 20 FPS in about 6 GB, and verification decodes add roughly 10%. Each full pass over all datasets (~170k frames, subsampled with the standard eval strides) costs about 2-3 GPU-hours. The ~25 method/ablation/budget configurations then come to about 70 GPU-hours. Baselines (Scal3R, MASt3R/hloc, optional StreamVGGT/Point3R on subsets) add about 25 GPU-hours, and descriptors plus the week-1 oracle study about 20 GPU-hours. The bank, FAISS and GTSAM all run on the CPU.

**Risks.**

1. Frame dependence (main risk). Anchor3R shows that cached pose-query tokens hurt when reused across coordinate frames. Blending state from two differently drifted frames may cause pose jumps or duplicated geometry. Mitigations: the R2 hard switch with frame re-anchoring, restoring only image-conditioned tokens, and read-only verification before any restore.

2. Overlap with Scal3R. Scal3R already shares the backbone and the retrieval stack and may capture most of the within-session ATE gain. That is why cross-session warm start is the headline and why the image-re-processing control is mandatory.

3. Snapshot quality. The CUT3R state may be a poor relocalizer when the viewpoint changes a lot. In that case, the week-1 kill test would show it.

4. Perceptual aliasing. Aliasing on KITTI and in repetitive rooms can trigger false restores. Consecutive-frame verification and false-loop-rate reporting address this.

5. RIO10 access. The data is only available on request, and its test GT is hidden, so only about 10 train/val pairs are usable. Fallbacks are 3RScan rescans with the provided alignments, or 7-Scenes/12-Scenes only.

6. A crowded, fast-moving field. Several related arXiv papers appeared in the last 3 months, so the cross-session result should be posted to arXiv early.

7. Unverified details. The web-search budget ran out during planning. Code availability for TTSA3R, MeMix, FILT3R, Info3R and ReCal3R was not re-verified; treat them as optional baselines. The exact layout of CUT3R's pose-retriever memory should be confirmed in the code. Scal3R's algorithm should be re-checked once arXiv is reachable, to confirm it never restores the full decoder state.

**Target venue.** ICCV 2027 (deadline around March 2027); fallback IROS 2027 / RA-L, or a 3DV 2027 / CVPR 2027 workshop for an early short version

**First-round prior-work verdict: partially_novel.** Retrieving places with SALAD+FAISS on top of streaming 3R models and closing loops with a pose graph already exists (2608.27529, HorizonStream). Scal3R does this on the same frozen CUT3R backbone: its README confirms DINOv2-SALAD/FAISS loop closure with GTSAM and a CUT3R state reset every 10 frames. It also re-injects archived pose tokens, but the exact contents of its archive could not be checked because arXiv was unreachable. The defensible delta has three parts: (1) training-free restoration of the full recurrent scene state itself, after resets and on revisits; (2) cross-session warm start of that state with change-gated release on RIO10, for which no prior work was found; and (3) a controlled test, using identical retrievals, of stored state against stored images, pose tokens and pose-graph-only correction.

**Closest work found in the first round:**

- [Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction (arXiv 2609.04201; NVlabs/scal3r)](https://arxiv.org/abs/2609.04201) — arXiv 2026-09 / ECCV 2026 poster. This is the most threatening work. Its frozen streaming decoder state is reset every N_reset frames. Non-keyframe KV state is discarded by restoring a pre-forward snapshot, so state snapshot/restore exists here but only as one-step rollback. Loop closure runs DINOv2-SALAD with FAISS over archived keyframe descriptors and re-injects the archived camera token of the matched keyframe into the pose-token buffer as an extra reference slot. An iSAM2 factor graph then corrects poses, which are written back into the buffer. This covers place-indexed archive, retrieval, re-injection of stored tokens and pose-graph correction. Differences from the idea: it archives camera/pose tokens, not the full recurrent scene state; it relies on learned pose tokens (~1% params) rather than being training-free; and it has no cross-session warm start. A GitHub issue indicates CUT3R-with-reset comparisons. I could not open the page because WebFetch failed with DNS errors, so this account comes from search snippets.
- [Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction (arXiv 2608.27529)](https://arxiv.org/pdf/2608.27529) — arXiv 2026-08. Adds an optional training-free loop-closure backend: FAISS retrieval over DINOv2-SALAD descriptors, pairwise relative pose on local windows, then sparse pose-graph optimization. It explicitly notes that CUT3R/Point3R absorb revisits as ordinary memory writes, so drift accumulates. This matches the idea's own 'key baseline' (retrieval plus pose graph). It does not restore the recurrent state.
- [Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction (arXiv 2605.05749)](https://arxiv.org/html/2605.05749v3) — arXiv 2026-05. Detects loop revisits from the memory itself, using spatial/angular/temporal thresholds on pointers. It adds pose constraints and pose-graph refinement, then rewrites the pointer memory in the corrected frame. This is memory-level loop handling, but for a Point3R-style explicit pointer memory, not restoration of a stored CUT3R/TTT3R state snapshot.
- [Info3R: Information-Adaptive Test-Time Training for 3D Reconstruction (arXiv 2609.21938)](https://arxiv.org/html/2609.21938) — arXiv 2026-09. A TTT3R-style information-aware learning rate plus a Dynamic State Reset that re-initializes a saturated state when a trigger metric exceeds a threshold, with anchor-to-world alignment. It covers reset timing but discards the state; no stored past state is restored.
- [Adaptive World Memory 3D Foundation Model for Scalable 3D Mapping, Localization, and Rendering (arXiv 2609.21502)](https://arxiv.org/abs/2609.21502) — arXiv 2026-09. A memory-centric 3D foundation model with gated recurrent memory updates plus test-time temporal-spatial masks. It splits memory into submaps, each with its own local memory, and aligns them through loop closure and SL(4) global optimization. It also targets localization. Keeping per-submap memories is close to keeping multiple state banks, but it is a trained model and does not retrieve or blend old states into the live state.
- [Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping (arXiv 2606.05035)](https://arxiv.org/pdf/2606.05035) — arXiv 2026-06. Its ablation reports that caching image-conditioned states is safe but caching pose-query tokens hurts later predictions, because those tokens depend on a local coordinate frame. This bears directly on the idea's risk that restored or blended tokens from a differently drifted frame may be inconsistent. It is evidence on the idea's main risk, not a scoop.
- [ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction (arXiv 2607.05356)](https://arxiv.org/pdf/2607.05356) — arXiv 2026-07. A training-free per-token reliability-calibrated learning rate on CUT3R that protects reliable state tokens from being overwritten (3.7x ATE reduction). Like MeMix, FILT3R, TTSA3R and AFG (2605.16981), it changes how the state is overwritten and does not recover overwritten state.
- [HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction (arXiv 2605.23889)](https://arxiv.org/pdf/2605.23889) — arXiv 2026-05. An optional loop-closure module retrieves revisited frame pairs from stored early-layer DINOv2 features and passes them back through the network for local corrections, which then feed a pose graph. This is retrieval plus re-processing, not state restoration.
- [STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction (arXiv 2603.20284)](https://arxiv.org/html/2603.20284) — arXiv 2026-03. Keeps evicted tokens in a voxel grid for later spatial retrieval. This is an external store of evicted memory with place-indexed recall, but for KV-cache transformers, not recurrent-state snapshots.

---

## 3. Keep the Address, Evict the Text: Cued Self-Test Eviction for Facts Hidden in a Small LM's Weights

*First-round title: Evict the Text, Keep the Address: Self-Test-Gated Eviction Turns a Retrieval Store into an Index over a Small LM's Weights*

**Judge:** overall 6.5/10, tier A. Scores out of 5: novelty 3, low compute 4, impact 4, speed of first signal 4.
This plan targets the verified open problem in O'Neill 2607.11020: forgotten facts keep their log-prob, wrong answers redirect to the latest write, and only full in-context text restores them. Its proposed third store state, an address-only entry, goes straight at the doc's question of what the external store should keep once content has moved into the weights. The pilot costs 10-15 GPU-hours on Qwen 0.5B, and even a negative result (that addresses act only as generic priming) is informative. Its novelty survives Dual-Layer only once the claim is narrowed to the address state plus the delayed, cued gate.

**Biggest weakness:** Several prior results point toward a null: Pseudo-Forgetting suggests generic or optimized suffixes may recover as much as a correct address, REMIX finds random codes are the most forgettable, and O'Neill reports anti-savings. Disentangling the template confound is subtle, and Dual-Layer's group can easily extend into this space.

**One line.** After a fact is fine-tuned into a small LM, the retrieval store replaces its text with a 4-token address that re-cues the weight copy. The text is deleted only after a delayed cued self-test passes. The method is scored on retention per stored token and per context token against full-text RAG, LLMLingua-2 compression and replay.

**Memory store.** Cross-store consolidation: external store to weights. After a write, the store keeps only an address (index entry) into the weights, and a self-test gates eviction of the text.

**Task.** Sequential injection of fictional facts into a small base LM. The stream is 20 write episodes of bioS-style biographies, each with 50 new people x 4 attributes = 200 atomic facts and 8 template restatements per fact. About 10% of later facts overwrite earlier ones. Evaluation uses direct, held-out-template paraphrase and 2-hop questions over all past facts. Secondary checks: the kaist-ai/fictional-knowledge probes (memorization, semantic, compositional) and a small EvolvingQA update stream.

**Gap in the big picture.** Lesson 3 of the big-picture doc: with a finite context, old and new items compete at retrieval (2604.27003), and text memories that keep being rewritten degrade (2605.12978). It also addresses the open consolidation problem ('what should move from the store into the weights, and when'), specifically the unmeasured second half: when can the store forget? Consolidating a fact into the weights saves context and storage only if its text can leave the store. No current protocol, including Harrington et al.'s 2607.07847 sequential protocol, tests whether a consolidated fact stays reachable once its text is gone. The work builds on the reported finding (2607.11020, unverified this session) that later writes hide earlier facts rather than erase them, and that only re-supplying the full fact text recovers them.

**Hypothesis.** Setup: a 20-episode fictional-biography stream on Qwen2.5-1.5B with a fixed 256-token retrieval context that includes same-subject and same-relation distractors. Prediction: self-test-gated address-only memory keeps at least 90% of unlimited full-text-RAG accuracy on paraphrased questions about episode 1-10 facts while holding at most 10% of full-text RAG's hot-store tokens. It also beats budget-matched full-text RAG, LLMLingua-2-compressed text and 10% experience replay at equal context tokens and equal training compute.

**Method.**

Write: each episode fully fine-tunes the model (LoRA r=16 as an ablation) on that episode's restatements plus QA-format examples for half the people (the Physics-of-LMs mixed-training recipe, so facts can be extracted). Every restatement of fact f carries an address a_f. Address types are (i) a natural-language subject+relation stub; (ii) a random 4-token code built from 256 reserved rare existing vocabulary tokens, with pairwise Hamming distance >= 2 and no shared prefixes, so no embedding growth; (iii) a per-episode soft prompt of 1-4 vectors fitted after the write and then frozen. Ablation over address granularity: per fact, per entity, per episode. The per-episode version is the MeCo/source-tag analogue. Store: each fact is an entry with key = a Contriever or BM25 embedding of subject+relation, value = a_f, plus the text while not yet evicted. Eviction gate: d episodes after the write (d in {0,1,3}), fact f is self-tested with its address on m=3 probe paraphrases. The probe templates are disjoint from the evaluation templates, or are LLM-generated. The text is deleted iff cued accuracy >= tau (default 2/3). Facts that fail keep their text and can get a targeted 'refresh' rehearsal, with the address, in the next write, which is replay restricted to failing items. Query: the retriever fills a fixed T-token context with top-k addresses and any surviving texts. For 2-hop questions the model first recites each addressed fact ('recite-then-answer') and then composes. Because the gate decisions do not change the weights (except in the refresh variant), all eviction policies (never, immediate, age/LRU, random at matched rate, self-test-gated, oracle) are replayed offline from per-fact logged cued-recall trajectories, almost for free. Diagnostics: recovery ratio R = (acc_cue - acc_nocue)/(acc_fulltext - acc_nocue) on hidden facts. Validity controls: a shuffled code from the same episode, a generic training-format prime, and the same address prepended to the never-trained base model. Cue recovery is correlated (Spearman) with relearning savings to validate it as a cheap hidden-vs-erased probe.

**Datasets.**

- Synthetic bioS-style fictional biographies (own generator following Allen-Zhu & Li, Physics of LMs 3.1): 1,000 people x 4 attributes over 20 episodes, 8 restatement templates, held-out paraphrase templates, 2-hop links (e.g. employer -> employer's HQ city across episodes), 10% overwrite facts, same-name/same-relation distractors
- kaist-ai/fictional-knowledge (Chang et al. 2024, 'How Do LLMs Acquire Factual Knowledge During Pretraining?'): memorization, semantic-generalization and compositional probes (HF availability not re-verified this session)
- EvolvingQA (Kim et al., 'Carpe Diem' continual knowledge benchmark): small update-stream subset, secondary real-world check
- Self-test paraphrase pool generated by an API LLM or held-out templates, kept disjoint from the evaluation templates

**Models / checkpoints.**

- Qwen/Qwen2.5-0.5B (base; main pilot model)
- Qwen/Qwen2.5-1.5B (base; main headline model)
- meta-llama/Llama-3.2-1B (base; cross-family check)
- facebook/contriever-msmarco (dense retriever for store keys)
- BM25 via rank_bm25 (sparse retriever; PyPI package verified)
- microsoft/llmlingua-2-xlm-roberta-large-meetingbank (LLMLingua-2 compressor baseline; llmlingua 0.2.2 on PyPI verified)
- Optional: Qwen/Qwen3-0.6B-Base as a newer-family check

**Baselines.**

- Weights only, no cue (sequential fine-tuning, question only)
- Full-text RAG with no eviction and unlimited context (accuracy upper bound, maximum tokens)
- Full-text RAG under the same fixed T-token / k-entry budget with distractors (retrieval-competition baseline from 2604.27003)
- LLMLingua-2-compressed fact text at matched context tokens
- Per-fact soft-token knowledge carrier fitted on the frozen base model (gist/Cartridges-style compact substitute; stands in for xRAG, whose released projectors target 7B models, not Qwen-0.5B/1.5B)
- Experience replay with a 10% text buffer, and replay matched to the refresh variant's replayed tokens
- Shuffled-address and generic training-format-prime controls (separate addressing from task-alignment restoration, as in Spurious Forgetting)
- Eviction policies: never evict, evict immediately, age/LRU, random at matched eviction rate, oracle (evict iff the fact is later recoverable)
- Episode-level source tag (MeCo / source-aware-training analogue) versus per-fact address

**Metrics.**

- Exact match and token-F1 on direct, paraphrased and 2-hop questions over all past facts after each episode; stream-average accuracy and backward transfer
- Hidden-fact recovery ratio R = (acc_cue - acc_nocue) / (acc_fulltext - acc_nocue)
- Hot-store tokens and bytes per fact; context tokens per query
- Accuracy under a fixed context budget (T = 128/256/512 tokens) with distractors
- Pareto AUC of retention against stored tokens, and retention against context tokens
- Eviction regret rate (evicted facts that later fail cued recall and are unrecoverable) and false-keep rate
- Redirection rate (errors that output another fact's value for the same relation), compared by address type
- Spearman correlation between cue recovery and relearning savings (validates the hidden-vs-erased probe)
- Training GPU-seconds per episode (equal-compute comparison with replay)

**First experiment (go/no-go).**

Weeks 1-2 on one GPU (A100 or a 24 GB card), Qwen2.5-0.5B, 10 episodes x 200 facts, 2 seeds. Three write conditions: no address, NL stub, random 4-token code; each episode is about 1-2 minutes of full fine-tuning. After episode 10, build the 'hidden' set: facts from episodes 1-3 that fail no-cue but pass with the full text in context. On it, measure cued accuracy on held-out paraphrases for (a) the correct code, (b) the stub, (c) a shuffled code from the same episode, (d) a generic training-format prime, and (e) the correct code on the never-trained base model. Also measure recite-then-answer 2-hop accuracy and the redirection rate. Then replay the eviction gate offline (d=1, tau=2/3) and compute regret on episodes 4-10 versus never-evict and evict-immediately. GO if: the code reaches R >= 0.5; the shuffled code and format prime reach R <= 0.15; the code beats the stub by >= 5 points or matches it with fewer redirections; and the gate's regret is < 20% at >= 5x hot-store reduction. Total pilot cost is about 10-15 GPU-hours.

**Kill criterion.**

Drop the address-only store if, in the pilot, correct-code cues recover R < 0.3 of the full-text gap on hidden facts, or a shuffled code or generic format prime recovers within 10 points of the correct code. Either result would mean the effect is generic task-alignment priming rather than addressing. Drop the headline economics claim if, at full scale on Qwen2.5-1.5B, either of two things happens. First, self-test-gated eviction has > 20% regret at the threshold that gives a 10x hot-store reduction. Second, its retention-per-token Pareto curve is dominated by LLMLingua-2 compression or by replay at matched training compute. In that case the fallback is a short diagnostic paper on cue recovery as a hidden-vs-erased probe, provided it correlates with relearning savings (Spearman >= 0.5); otherwise abandon.

**Compute.** 150 GPU-hours over 12 weeks. 1x NVIDIA A100 80GB (or H100) for full fine-tuning of the 1.5B model. The 0.5B/1B runs, LoRA ablations and all evaluation (vLLM) also fit on a single 24 GB RTX 4090/A5000. Retrieval, LLMLingua-2 compression and offline eviction-policy replay run on CPU or alongside on the same card. Self-test paraphrase generation uses an API LLM (low cost) or templates.

**Risks.**

(1) Novelty is not fully verified. Both passes ran without web search, and the premise paper 2607.11020 could not be opened, so its 77-80% full-text recovery figure is unverified. Concurrent 2026 work on consolidating RAG stores into weights, or on 'sleep' agents, is the main scoop risk. (2) Cue recovery may be generic re-alignment rather than addressing, as Spurious Forgetting would predict. The shuffled-code, format-prime and base-model controls exist to catch this, and failing them kills the main claim. (3) Addresses can be overwritten or collide too; DSI++ shows docid forgetting under continual indexing. Code design (Hamming distance, no shared prefixes) and the measured redirection rate address this. (4) Facts may become recallable only with the tag. That is acceptable if the store always supplies it, but it must be reported, and a lost address means a lost fact (measured by regret). (5) Gains may hold for recitation but not composition. 2-hop and recite-then-answer are tested from week 1. (6) Leakage between self-test and evaluation paraphrases would inflate the gate, so template pools and LLM-paraphrase sources are kept disjoint. (7) With synthetic facts, retrieval may be too easy for the 2604.27003 retrieval-competition effect to appear, so distractors must be deliberately hard (same name, same relation). (8) LoRA may hide facts differently from full fine-tuning, so it is reported as a separate ablation. (9) 0.5-1.5B models are small, so it is unclear whether the effect holds at 7B. That is out of budget and stated as a limitation.

**Target venue.** ICML 2027 (late-January 2027 deadline). Fallbacks: ACL 2027 via ARR (Feb 2027), COLM 2027, or CoLLAs 2027.

**First-round prior-work verdict: partially_novel.** Each ingredient exists separately: training facts with attached IDs (source-aware training, MeCo, DSI/DSI++); key-retrieved soft prompts that unlock model behaviour (L2P/Progressive Prompts); 'forgetting is hidden, not erased' (Spurious Forgetting, anticipatory recovery, and the cited 2607.11020); and compact context substitutes (gist tokens, Cartridges). The defensible delta is the store-management policy and its economics: deleting consolidated text, keeping only an address into the weights, gating deletion with delayed cued self-tests, and scoring retention per stored and per context token under retrieval competition. A secondary contribution is cue recovery as a cheap hidden-vs-erased probe, validated against relearning savings. Caveat: neither this pass nor the novelty pass could search or open papers (web-search budget exhausted, arXiv and Hugging Face blocked). 2607.11020 and any 2026 'RAG-to-weights consolidation with eviction' or sleep-consolidation work must be checked before committing.

**Closest work found in the first round:**

- [Source-Aware Training Enables Knowledge Attribution in Language Models (Khalifa et al.)](https://arxiv.org/abs/2404.01019) — COLM 2024 (arXiv 2404.01019). Trains a small LM on synthetic fictional-biography documents, each tagged with a unique document identifier injected into the training text, so that facts and short IDs are bound in the weights. The direction is reversed (the model outputs the ID for a fact, used for citation), there is no continual stream, and the ID is never used as an input cue to recover a fact. Still, it shows that training on ID-tagged facts is established practice, so the 'random code address' component is not new on its own. Recalled from prior knowledge. The fetch failed (arxiv DNS/proxy blocked), so details are unverified this session.
- [Spurious Forgetting in Continual Learning of Language Models (Zheng et al.)](https://arxiv.org/abs/2501.13453) — ICLR 2025 (arXiv 2501.13453). Shows that much of the performance loss after sequential fine-tuning is lost task alignment, not lost knowledge, and that the knowledge can be recovered. This is the same 'hidden, not erased' premise the idea builds on (the idea attributes it to 2607.11020). It does not propose short address cues or an external store that becomes an index. Recalled from memory. The fetch was blocked.
- [Metadata Conditioning Accelerates Language Model Pre-training (MeCo, Gao et al.)](https://arxiv.org/abs/2501.01945) — arXiv 2025 (2501.01945). Prepends short metadata (source URLs) to training documents, then conditions on that metadata at inference to steer the model. This matches the idea's 'every training restatement carries an address; prepend the address at query time to reactivate' at pretraining scale. It does not target continual fact injection, eviction, or hidden-fact recovery. Recalled from memory and not opened.
- [Learning to Prompt for Continual Learning (L2P, Wang et al.) / Progressive Prompts (Razdaibiedina et al.)](https://arxiv.org/abs/2112.08654) — CVPR 2022 / ICLR 2023. Keeps a pool of key-to-soft-prompt pairs per task and selects a prompt by query-key match at test time, to unlock task-specific behaviour without replay. This is structurally the same as the idea's address type (iii): a per-episode soft prompt of 1-4 vectors fitted and frozen, fetched by a retrieval key. The differences are that L2P freezes the backbone and works on task/class CL, not facts written into trained weights. Not opened.
- [Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training (Yang et al.)](https://arxiv.org/abs/2403.09613) — NeurIPS 2024 (arXiv 2403.09613). Shows that LLMs fine-tuned on cyclic document sequences recover forgotten documents before seeing them again, more evidence that interference hides knowledge rather than erasing it. It has no cue/address mechanism and no store eviction. Not opened.
- [DSI++: Updating Transformer Memory with New Documents (Mehta et al.) and DSI (Tay et al.)](https://arxiv.org/abs/2212.09744) — EMNLP 2023 / NeurIPS 2022. Continual indexing of new documents with memorized docids. It studies forgetting of earlier docids during incremental writes and mitigates it (sharpness-aware training, generative memory replay). The idea already cites DSI as the reverse direction. DSI++ adds the continual-forgetting angle, which overlaps the claim that 'random codes resist redirection'. Not opened.
- [Cartridges: lightweight long-context representations via self-study (Eyuboglu et al.)](https://arxiv.org/abs/2506.06266) — arXiv 2025 (2506.06266). Replaces a corpus in context with a small trained KV-cache/prefix that is stored and loaded at query time. Like the soft-address variant, a compact learned artifact stands in for the text, but the knowledge sits in the prefix rather than in base weights that the prefix 'unlocks'. Not opened.
- [Learning to Compress Prompts with Gist Tokens (Mu et al.) / xRAG (Cheng et al.)](https://arxiv.org/abs/2304.08467) — NeurIPS 2023 / NeurIPS 2024. Compresses context into one or a few tokens. The idea already names xRAG as a baseline. Both show token-budget savings from compact cues, but neither uses the cue as a pointer to knowledge consolidated into the weights. Not opened.

### Second-round re-check (live search)

**Revised one line.** After facts are fine-tuned into a <=1.5B LM, each store entry is downgraded from full text to a short address that re-cues the hidden weight copy. Text is deleted only when a delayed, address-cued self-test passes. The result is compared at matched hot-store and context tokens against Dual-Layer-style no-cue pruning, full-text RAG, LLMLingua-2 and replay.

**Verdict: partially_novel; recommendation: pursue_with_pivot.**

**What is taken and what is still new.** The plan is not scooped, but its framing is weaker than it claims. Dual-Layer Agentic Memory (2608.22215, Aug 2026) already consolidates external memories into a small LM's weights by SFT and prunes the store (up to 68% removed at more than 98% EM). Its gate is no-cue answerability, so the store keeps either the text or nothing. EVAF (2606.29916) already gates parametric consolidation with a test-retest harness. The claims 'nobody measures whether a consolidated fact survives once its text is gone' and 'self-test gating is new' should both be dropped. What remains new, as far as 12 searches show, has three parts. (1) A third store state between keeping the text and evicting it: keep only a short ADDRESS that re-cues a fact that is hidden in the weights but not reachable. This targets exactly the regime 2607.11020 documents (log-prob retained, redirection to recent facts, 77-80% recovered only with the full text in context, no weights-only fix) and its open problem. (2) A DELAYED, address-cued self-test as the eviction gate, run after later writes have interfered, compared against Dual-Layer's no-cue gate. (3) A controlled comparison of address types (random code, NL stub, soft prompt) against shuffled-code, format-prime and optimized-suffix controls, with retention per hot-store token and per context token under retrieval competition (2604.27003). No search found per-fact ID cues used as input pointers to recover facts after sequential fine-tuning. REMIX (2411.07175) suggests random keys are the most forgettable, so the random-code arm may well lose to NL stubs. That outcome is still publishable inside the comparison.

**Required revisions.** 1. Citations. 2607.11020 is verified: Charles O'Neill (Baseten), Jul 2026, Qwen3 models; it reports 77-80% recovery with the fact in context and redirection errors to the latest fact. Mark it verified and fix the attribution. 2604.27003 is also verified ('When Continual Learning Moves to Memory', Apr 2026). 'Harrington et al. 2607.07847' could NOT be found by ID, author or topic. Remove it, or verify it by hand before citing; do not build the gap statement on it. 2. Premise tension. 2607.11020 also reports SLOW relearning (anti-savings), which argues against 'merely hidden'. It ran a cued-question probe but excluded it for template confound. The plan's relearning-savings validation should therefore expect weak savings: report the cue-recovery vs relearning correlation as an open empirical question, not as a validation step, and keep self-test, evaluation and training templates strictly disjoint. Consider adding Qwen3-0.6B/1.7B-Base as main models so the setup is directly comparable with 2607.11020. 3. Rewrite the gap statement. Dual-Layer Agentic Memory (2608.22215) already does store-to-weights consolidation plus store pruning at near-full EM. The contribution is not 'consolidate then evict'. It is (a) the address-only intermediate state for facts that are hidden, not erased; (b) a delayed cued gate against an immediate or no-cue gate; (c) per-token Pareto economics under retrieval competition. Drop the claim that no protocol measures post-eviction reachability. 4. Self-test gating. Cite EVAF (2606.29916) and Memory Depth (2606.26806) as prior test-retest and consolidation-gating work, and narrow the claim to an address-cued, delayed gate for text eviction. 5. Random codes. REMIX (2411.07175) finds random key strings are the most forgettable. Pre-register that NL stubs may win, and report code-binding survival per episode. 6. Controls. Pseudo-Forgetting (2411.11932) shows meaningless or GCG suffixes reactivate hidden ability, so add an optimized-generic-suffix control next to the shuffled-code and format-prime controls. 7. Cut scope to stay under about 150 GPU-hours: drop the EvolvingQA arm unless the pilot is GO, and keep Llama-3.2-1B as a single-seed check.

**Baselines to add.**

- Dual-Layer-style no-cue pruning gate (2608.22215): after SFT write-back, evict the text iff the model answers with NO cue; keep the full text otherwise (no address state)
- Immediate cued-gate vs delayed cued-gate (d=0 vs d>=1), to isolate the value of re-testing after interference
- TokMem-style per-fact dedicated memory token trained with the backbone frozen (2510.00444), as a parameter-isolated alternative to an address into shared weights
- Optimized generic suffix (GCG or a soft prompt trained on a disjoint fact set) as a non-addressing recovery control (per Pseudo Forgetting, 2411.11932)
- EVAF-style surprise-gated LoRA consolidation (2606.29916) as an alternative write-selection policy, if cheap to reproduce
- Auto-Dreamer-style store-only compaction (summarise/merge entries, no weight writes) at matched hot-store tokens (2605.20616)

**Closest work (second round):**

- [Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation (Wenzhi Li et al.)](https://arxiv.org/pdf/2608.22215) — arXiv Aug 2026 (2608.22215). The closest scoop risk. A fast router labels incoming knowledge non-write / write-new / write-update, depending on whether the model already answers correctly without memory. A slow write-back then moves high-value external memories into the weights by SFT. A 1.7B/8B cascade reportedly prunes up to 68% of redundant external memory while keeping more than 98% of store-everything QA EM. This already covers 'consolidate into weights, then shrink the store, scored by retention against store size'. Its pruning gate is no-cue answerability, and the store keeps either full text or nothing. As far as the snippets show, it has no address or pointer entries, no delayed re-test under later sequential writes, and no hidden-vs-erased analysis. Known from search snippets only.
- [Can a Language Model Learn Facts Continually in Its Weights? (Charles O'Neill, Baseten)](https://arxiv.org/pdf/2607.11020) — arXiv Jul 2026 (2607.11020). This is the plan's premise paper, and its ID is now VERIFIED. It writes invented facts into Qwen3 models sequentially. After 20 writes, retention is 1% for bare statements and 46% for broad restatements. Forgotten facts keep most of the log-prob their write added, and wrong answers mostly name the most recently written fact (redirection). Supplying the fact in context restores 77-80%. No intervention kept facts reachable through weights alone. Caveats that matter for the plan: (a) it reports SLOW relearning, the opposite of savings, which argues against 'merely hidden'; (b) it ran a cued-question re-test (study 3/3 vs bare 0/54) but excluded it because the cue questions share their contrast form with the training data. The plan's cue-address idea directly answers this paper's open problem, but it must avoid the same template confound.
- [EVAF: A Test-Retest Protocol for Selective Parametric Consolidation](https://arxiv.org/pdf/2606.29916) — arXiv Jun 2026 (2606.29916). A dual sigmoid write gate (surprise/valence) for LoRA consolidation, plus a falsifiable test-retest harness that detects parametric writes, with replay-retention (anti-forgetting) checks. 'Test-retest gating of what was consolidated' therefore already exists. The domain differs (persona and goal tendencies, not atomic facts), and the protocol is not used to delete text from an external store.
- [Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents (Haoliang Han)](https://arxiv.org/pdf/2606.26806) — arXiv Jun 2026 (2606.26806). Compares a retrieval index with selective parametric consolidation (EVAF LoRA) on GPT-2/TinyLlama/Mistral-7B. Retrieval wins on short-fact recall (0.956-0.973), while consolidation wins on persistence after context unload. It names stale-memory invalidation and delete/update validity as unresolved. This is the same store-vs-weights framing on small models, but the retrieval index is kept intact, with no eviction and no address.
- [Continual Memorization of Factoids in Language Models (REMIX)](https://arxiv.org/html/2411.07175) — arXiv Nov 2024 (2411.07175). Studies continual memorization of key-value factoids under later fine-tuning and finds that randomly generated key-value strings are the MOST susceptible to forgetting. This directly threatens the plan's random 4-token code address: the code-to-fact binding may be the first thing a later write erases. It also has a selective-recall probe given two keys.
- [TokMem: One-Token Procedural Memory for Large Language Models](https://arxiv.org/pdf/2510.00444) — arXiv Oct 2025 (2510.00444). Gives each memory item (a procedure) its own dedicated token, parameter-isolated from the backbone, so items can be added without interference. This is mechanistically the nearest thing to a per-item address token. The differences: it stores skills, the knowledge lives in the token rather than in shared weights it points into, and there is no text eviction.
- [When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents](https://www.emergentmind.com/papers/2604.27003) — arXiv Apr 2026 (2604.27003). Cited ID VERIFIED. Shows that with a finite context, old and new experiences compete at retrieval (ALFWorld/BabyAI), so external memory reshapes the continual-learning problem rather than solving it. This motivates the fixed-budget retrieval baseline. It does not consolidate into weights.
- [Unveiling and Addressing Pseudo Forgetting in Large Language Models](https://arxiv.org/pdf/2411.11932) — arXiv Nov 2024 (2411.11932). Shows that tasks 'forgotten' after sequential learning can be reactivated by semantically meaningless suffixes or GCG-optimized prompts. Generic prompts can therefore recover hidden capability, which makes the plan's shuffled-code and generic-prime controls essential. An optimized-suffix control should be added.
- [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](https://arxiv.org/pdf/2605.20616) — arXiv May 2026 (2605.20616). A learned offline consolidator (GRPO) that replaces regions of an external memory bank with compact abstractions, using a 12x smaller active bank. This is store-to-store consolidation and pruning, not store-to-weights, and it should be cited for the 'shrink the store' economics.

**Queries run:** arXiv 2607.11020; consolidate RAG memory into LLM weights then evict text from external store 2026 (extended); arXiv 2607.07847 sequential knowledge injection protocol Harrington; EVAF test-retest protocol selective parametric consolidation; fine-tuned LLM forgotten facts recovered by cue prefix hidden not erased sequential fine-tuning 2026; hippocampal indexing theory language model external memory stores pointers to parametric knowledge index tokens; "Can a Language Model Learn Facts Continually in Its Weights?"; "Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation"; arXiv 2604.27003 external memory finite context old and new memories compete retrieval continual learning; language model trained on facts tagged with identifier tokens, prompting with the identifier recalls the fact after continual fine-tuning (extended); "Memory Depth, Not Memory Access" selective parametric consolidation long-running language agents; Auto-Dreamer 2605.20616 offline consolidation external memory into weights prune memory after consolidation

---

## 4. Rewrite or Advance? Compute-Matched State-Writing Replay for Recurrent Streaming 3D Reconstruction

**Judge:** overall 6/10, tier B. Scores out of 5: novelty 3, low compute 4, impact 3, speed of first signal 4.
It brings the classic continual-learning lesson (replay is the bar to beat, and compute rather than storage is the budget) into a recurrent 3D fast state, with an equal-pass frontier and stacking on 2026 update rules. The patch is about 100 lines, the pilot costs about 8 GPU-hours, and it shares the TTT3R fork, data and evaluation with P1. It is the natural second chapter of the 3D arc, covering the data buffer to fast state direction.

**Biggest weakness:** Replaying old keyframes into a state that tracks the current pose frame may simply hurt, and any gains may appear only on loop sequences, where Anchor3R/ABot-Recon-style reinsertion already covers the ground. Gains may also vanish once stacked on ReCal3R or TTSA3R.

**One line.** A CUT3R/TTT3R-style reconstructor has a fixed number of forward passes per second of video. A training-free scheduler gives 5-50% of those passes to re-writing stored keyframes into the recurrent state rather than to new frames. The paper's main result is an accuracy-vs-passes frontier that tests whether this beats, and stacks with, every equal-compute state-update rule.

**Memory store.** Data buffer (CPU keyframe buffer of images or cached encoder tokens with predicted poses) consolidated into the fast state (CUT3R's 768 recurrent state tokens) through replay-as-write.

**Task.** Real-time streaming 3D reconstruction under a fixed forward-pass budget P passes per second of 30 FPS video, with P in {15, 7.5, 3.75}, on long sequences of 1000-4500 raw frames. Pose is evaluated on ScanNet, TUM RGB-D and KITTI odometry, dense reconstruction on 7-Scenes and Neural-RGBD, and video depth on Bonn and KITTI. Results are reported separately for loop and exploration (non-loop) sequences.

**Gap in the big picture.** The plan rests on four points from the big-picture document.

1. Lesson 1: replay is still the bar to beat, and compute, not storage, is the real budget (Cho et al. 2502.07274). In streaming 3D the binding budget is forward passes per second, and every current method spends all of it on the newest frame.
2. The fast-state failure mode: fixed capacity, overwritten every step. TTSA3R names this catastrophic forgetting, and AFG puts the memory horizon at about 3 frames.
3. The doc's open problem: what should move between stores, and when. This project moves knowledge from the data buffer into the fast state.
4. Lesson 2: forgetting tracks how far an update moves the model. Here that motivates beta-scaled, gentle replay writes.

The 2026 training-free CUT3R literature (TTT3R, TTSA3R, AFG, Info3R, ReCal3R) only changes how strongly the newest frame writes. Online re-writing of old observations under a matched budget has not been studied.

**Hypothesis.** At 3.75-7.5 forward passes per second of 30 FPS video, replays of 10-25% of passes will cut median ATE by at least 15% compared with the best new-frame-only schedule at equal passes. Replays here are state-writing keyframe passes (beta > 0). The new-frame-only comparisons are oracle-tuned stride, adaptive keyframe selection, and AFG/Info3R-style gating. The claim covers sequences of at least 1000 raw frames, including non-loop exploration sequences. The gain is predicted to persist when replay is stacked on TTSA3R or ReCal3R.

**Method.**

1. Budget. A stream of N frames at 30 FPS and a budget of P passes per second give M = N*P/30 slots. Every method, baseline or not, gets exactly M passes. A replay is counted as a full pass; a FLOP-matched accounting, where cached ViT-L encoder tokens make replays cheaper, is reported separately.

2. Scheduler. Each slot takes either the newest frame or a replay from a CPU keyframe buffer, so a replay fraction rho lowers the new-frame rate to (1-rho)*P. Replays are added to CUT3R's view list, whose `_forward_impl` already has per-view update and reset masks. The state update for a replay is S <- S + beta * eta(x_k, S) * (S_new - S). Here eta is the per-token learning rate of the host rule (CUT3R, TTT3R's sigmoid of state-to-image cross-attention, TTSA3R or ReCal3R), and beta in {0, 0.25, 0.5, 1} scales replay writes.

3. Pose memory. The pose-retriever local memory (update_mask2) is frozen during replays, so an out-of-order view is not read as a pose jump. Out-of-order input is in distribution: CUT3R's training sampler (base_multiview_dataset.py) draws about 50% repeated-view sequences and 40% random-order sequences.

4. Selection policies:
   - anchor (frame 0)
   - round-robin oldest
   - reservoir
   - coverage-near: old in time but with high predicted frustum overlap with the current view
   - coverage-far: low overlap, meaning content about to be forgotten
   - forgetting-triggered: each replay's re-predicted pose is compared with its stored pose. That discrepancy is a free forgetting signal and is used to adapt rho online.

5. Deliverable. Accuracy-vs-passes frontiers (ATE, Acc/Comp, AbsRel against P) for rho in {0, 5, 10, 25, 50}%, with replay stacked on each host update rule. Results are split into loop and exploration sequences by a ground-truth revisit ratio.

6. Prior work as special cases. Setting beta = 0 and fusing the old-pose re-estimates in a small pose graph reproduces Anchor3R/ABot-Recon-style reinsertion. Replaying every frame reproduces CUT3R's offline revisit=2 at 2x compute.

7. Diagnostics. A read-only probe, run outside the budget, measures a forgetting curve of the state against lag for each update rule. Comparing anchor replay with recent or coverage replay tests two explanations of drift: forgetting of the anchor, versus first-frame gauge coupling as argued by LongStream and Anchor3R.

**Datasets.**

- ScanNet test scenes: raw stride-1 streams of up to 3000 frames, plus the scannet_s3_{50..1000} splits from the TTT3R/CUT3R preprocessing
- TUM RGB-D: the dynamics split (tum_s1_{50..1000}) plus long static sequences, e.g. fr3/long_office_household, fr2/desk, fr1/floor
- KITTI odometry 00-10: loop sequences 00/02/05/06/07/08/09 vs non-loop 01/03/04/10
- 7-Scenes (mv_recon protocol, Acc/Comp/NC)
- Neural-RGBD (NRGBD) (mv_recon protocol)
- Bonn RGB-D dynamic (video depth)
- KITTI depth (video depth)

**Models / checkpoints.**

- CUT3R cut3r_512_dpt_4_64.pth (github.com/CUT3R/CUT3R, Google Drive link in README)
- CUT3R cut3r_224_linear_4.pth (fast pilot sweeps)
- TTT3R: same CUT3R checkpoint with model_update_type=ttt3r (github.com/Inception3D/TTT3R)
- TTSA3R official code on the CUT3R checkpoint (github.com/anonus2357/ttsa3r)
- ReCal3R official code on the CUT3R checkpoint (github.com/Powertony102/ReCal3R)
- Point3R, explicit pointer-memory contrast where replay means re-insertion (github.com/YkiWu/Point3R, NeurIPS 2025)

**Baselines.**

- Uniform stride at matched passes (CUT3R, TTT3R)
- Oracle-tuned stride per sequence length at matched passes (upper bound of fixed-stride schedules)
- Adaptive keyframe selection, error- and momentum-based, on CUT3R (arXiv 2510.23928)
- AFG frame-level state-update gate (arXiv 2605.16981): reimplemented, since no public code was found
- Info3R information-adaptive LR plus dynamic reset (arXiv 2609.21938): reimplemented, since no public code was found
- TTT3R periodic state reset (reset_interval 50/100)
- TTSA3R (official code)
- ReCal3R (official code)
- Read-only replay (beta=0) plus pose-graph fusion: Anchor3R/ABot-Recon-style keyframe reinsertion as an ablation
- CUT3R offline revisit=2 (eval/relpose/launch.py --revisit 2): 2x-compute reference
- Point3R at matched passes, with and without re-insertion replay

**Metrics.**

- ATE RMSE after Sim(3) alignment (evo)
- RPE translation and rotation
- Acc / Comp / NC on 7-Scenes and NRGBD
- Video depth AbsRel and delta<1.25 on Bonn and KITTI
- Area under the ATE-vs-passes-per-second curve (frontier dominance)
- Gain split by loop vs exploration sequences (ground-truth revisit ratio)
- State forgetting curve: read-only probe pose and depth error vs lag
- Replay discrepancy: stored vs re-predicted keyframe pose
- Wall-clock FPS, peak GPU memory, CPU buffer size, and FLOPs per pass type

**First experiment (go/no-go).**

Weeks 1-2, one 24 GB GPU, about 8 GPU-hours.

Setup:
- Clone TTT3R and download cut3r_512_dpt_4_64.pth.
- Write a ~100-line scheduler that interleaves replay views into the view list.
- In `_forward_impl`, multiply update_mask1 by beta for replay views and set update_mask2 = 0 for them.

Data, 16 sequences:
- 10 ScanNet test scenes at raw stride 1 with about 3000 frames each, split 5 loop-heavy and 5 exploration by ground-truth revisit ratio
- 4 long static TUM sequences
- KITTI 03 (no loop) and KITTI 00 (loop)

Conditions:
- Budgets: P = 7.5 and 3.75 (stride 4 and 8).
- Baselines: rho = 0 uniform stride, and oracle-tuned stride.
- Replay grid: rho in {10, 25}% x policy in {anchor, round-robin, coverage-near} x beta in {0, 0.5, 1}.
- About 40 configs in total.

Go signal: all three must hold.
1. Some beta > 0 replay config beats the tuned-stride baseline by at least 10% median ATE on the 1000+ raw-frame sequences.
2. That gain also appears on the exploration subset.
3. beta > 0 beats beta = 0, which shows that writing into the state matters beyond re-reading old poses.

The anchor-vs-recent contrast from the same runs gives the first reading on the drift hypothesis.

**Kill criterion.**

Drop or pivot if any of these holds:
1. At both budgets, no state-writing replay config (beta > 0) beats the oracle-tuned-stride or adaptive-keyframe baseline at equal passes by at least 5% median ATE.
2. Gains appear only on loop sequences and are matched by read-only replay plus pose-graph fusion. Replay would then reduce to Anchor3R/ABot-Recon reinsertion.
3. Gains fall below 5% once replay is stacked on ReCal3R or TTSA3R.

The fallback is a short diagnostic paper: forgetting curves of the recurrent state and the anchor-forgetting vs gauge-coupling test. If that is uninformative too, drop the project.

**Compute.** 150 GPU-hours over 12 weeks. 1x 24 GB GPU (RTX 4090, A5000 or L4 class). CUT3R at 512 px runs in about 6 GB at about 20 FPS, so a second GPU is optional and only parallelises sweeps. At least 32 GB CPU RAM for the keyframe buffer and data. No training at any stage.

**Risks.**

1. Replay may be equivalent to a smarter keyframe stride. Mitigation: oracle-tuned stride, adaptive keyframe selection and AFG/Info3R at exactly matched passes.

2. Replays may corrupt the state through out-of-order pose jumps. This is partly mitigated: CUT3R was trained with repeated and random-order views, the pose memory is frozen for replays, and beta scaling keeps replay writes small.

3. Gains may be concentrated on loop sequences, which collapses the idea into Anchor3R/ABot-Recon reinsertion. Mitigation: the explicit loop vs exploration split, with beta = 0 plus pose-graph fusion as a control.

4. AFG and Info3R have no public code that I could find. Reimplementations may be seen as weak baselines, so release the reimplementation code and reproduce their reported numbers first.

5. The area is crowded and fast-moving (about 7 training-free CUT3R update-rule papers in 2026), so a concurrent replay paper is plausible. Move fast and post on arXiv early.

6. The anchor-replay hypothesis may fail, since LongStream and Anchor3R argue that first-frame anchoring causes drift. This is framed as a decisive test that is publishable either way.

7. If FLOP-matched claims rely on cached encoder tokens, the cache costs about 1 MB per keyframe in fp16. Bounded buffers keep this small.

8. Novelty rests partly on search snippets. Read Info3R, AFG and Anchor3R in full in week 1.

**Target venue.** ICCV 2027 (deadline around March 2027). Fallback: 3DV 2027 or RA-L with an IROS 2027 option.

**First-round prior-work verdict: partially_novel.** Each nearby method covers part of the idea but leaves out the core:
- CUT3R's official code has an offline revisit mode (verified: `--revisit` and `--freeze_state` in eval/relpose/launch.py). It replays the whole sequence after the fact at about 2x compute, so it is not online and has no budget.
- Anchor3R (CoRL 2026) re-inserts retrieved loop keyframes, but its project page says this happens only in offline loop-aware refinement and only to add pose-graph edges in a 10-frame window/KV model. Its online path is strictly one-pass with "no loop-keyframe reinsertion".
- ABot-Recon spends extra passes on retrieved loop pairs.
- AFG, Info3R, TTSA3R and ReCal3R only change how strongly the newest frame writes.

The defensible delta has three parts: online state-writing replay into a recurrent fast state, a fixed pass budget with a matched-compute frontier, and stacking on the 2026 update rules. Read-only and pose-constraint replay are kept only as ablations attributed to Anchor3R/ABot-Recon.

Caveat: the web-search budget ran out during this step, and arXiv was blocked during the novelty check. Overlap claims rest on search snippets plus the repos and Anchor3R project page verified here. Read Info3R and AFG in full before committing.

**Closest work found in the first round:**

- [Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping (arXiv 2606.05035)](https://arxiv.org/html/2606.05035v1) — arXiv 2026 (June). This is the closest to the 'replay' operation. Search snippets show that historical or loop-closure keyframes 'can be reinserted into the active window' as transient anchors. Each reinsertion produces long-range relative-pose edges, and motion averaging then spreads drift over the pose graph. That is the same physical act of re-feeding stored keyframes. The differences: it is a window/KV model, not a recurrent fast state. Reinsertion appears to belong to an offline or refinement path, since its online variant is described as strict one-pass. The reinserted frames serve as pose-graph constraints, not as writes into a learned state, and there is no compute-matched slot accounting. It directly covers the idea's 'read-only replay used as a pose constraint' fallback and part of the 'free old-pose constraint' claim. It also argues that first-frame anchoring itself causes drift, which challenges the idea's 'anchor-replay recovers most of the gain' hypothesis.
- [Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction / ABot-Recon (arXiv 2608.27529)](https://arxiv.org/abs/2608.27529) — arXiv 2026 (Aug). It retrieves revisited historical frames with FAISS over DINOv2-SALAD descriptors. Local windows around the two frames are fed jointly into the model to estimate their relative pose, which yields loop-closure constraints for sparse pose-graph optimization. This is 'replay old keyframes through the network to get a fresh pose constraint'. The replay costs extra forward passes rather than being compute-matched, the model has no persistent recurrent state to rewrite, and it targets loop sequences. It overlaps with the idea's pose-smoother component and its S1/loop-sequence case.
- [Rethinking the State Update Gate for Long-Sequence Recurrent 3D Reconstruction (AFG, arXiv 2605.16981)](https://arxiv.org/pdf/2605.16981) — arXiv 2026 (May). It is training-free on TTT3R/CUT3R. A scalar frame-level gate, framed as a continuous relaxation of SLAM keyframe selection, down-weights redundant frames at no extra forward pass. It reports 51% lower ATE on long TUM sequences of up to about 4.6k frames. It diagnoses a memory horizon of about 3 frames, the same forgetting motivation. It never re-feeds old frames. It is the strongest equal-compute competitor and already takes the 'redundant frames waste the state' narrative.
- [Info3R: Information-Adaptive Test-Time Training for 3D Reconstruction (arXiv 2609.21938)](https://arxiv.org/abs/2609.21938) — arXiv 2026 (Sept 18). It is training-free on CUT3R/TTT3R. An importance score scales the state-update learning rate, which is redundancy-aware writing. A dynamic state reset is triggered by cumulative LR plus confidence, after which predictions are made in an anchor frame's coordinates. This covers the idea's 'drift-triggered' scheduling signal and its anchor-coordinate handling, but it resets the state rather than replaying. It is very recent and must be included as a baseline.
- [Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments (arXiv 2510.23928)](https://arxiv.org/abs/2510.23928) — arXiv 2025, ROBOVIS 2026. It plugs error-based (photometric/SSIM) and momentum-threshold keyframe selection into CUT3R and Spann3R. It beats fixed-interval and uniform-skip baselines while processing fewer frames. This is the 'smarter keyframe stride' control the idea fears is equivalent to replay. It selects new frames only and never replays old ones.
- [TTT3R: 3D Reconstruction as Test-Time Training (arXiv 2509.26645, ICLR 2026)](https://arxiv.org/pdf/2509.26645) — ICLR 2026. It provides the alignment-confidence learning rate the idea reuses. It also proposes periodic State Reset every 100 frames with chunks aligned by global poses, the classic alternative to replay for fixed-capacity state. No replay.
- [TTSA3R (2601.22615), MeMix (2603.15330), FILT3R (2603.18493), ReCal3R (2607.05356)](https://arxiv.org/abs/2601.22615) — arXiv 2026. A crowded family of training-free state-update rules for CUT3R: temporal-spatial adaptive updates, bottom-k patch writes, Kalman-style gain, and reliability-calibrated per-token LR (3.7x ATE reduction). All spend every forward pass on the newest frame. They are the 'training-free state-update rules' that the idea must beat or stack with. None replays.
- [LongStream (arXiv 2602.13172, CVPR 2026) and OVGGT / R^3 (2603.05959, 2605.26519)](https://arxiv.org/abs/2602.13172) — CVPR 2026 / arXiv 2026. These keep or protect keyframes. LongStream uses keyframe-relative poses plus periodic cache refresh. OVGGT uses a permanent first-frame anchor plus overlap-selected historical anchors. R^3 keeps a keyframe bank with novelty admission. They retain anchors in a KV cache instead of re-writing them into a recurrent state. LongStream argues that first-frame anchoring causes drift, which conflicts with the anchor-replay hypothesis.

---

## 5. Update Frequency Is Not Retention: Auditing and Fixing Cross-Session Consolidation in Continuum Memory Systems

*First-round title: Frequency Is Not a Timescale: When Do Multi-Frequency (CMS) Memories Actually Consolidate?*

**Judge:** overall 6/10, tier B. Scores out of 5: novelty 3, low compute 4, impact 4, speed of first signal 5.
It tests the doc's 'main bet' (HOPE-style multi-timescale memory) directly, and the CPU pilot is already done (the lemma holds to 3e-15 and decay produces an interior spacing optimum). An audit asking whether update frequency alone consolidates anything is high-visibility given the Nested Learning hype, and the Sleep paper (2606.03979) indirectly concedes the point. The cheap DeltaNet+MLP-CMS MQAR check can decide the paper within two weeks.

**Biggest weakness:** The principle is prior art (Jones et al. ICLR 2023, Smith 2006, Hinton & Plaut, Benna-Fusi, Mozer). Its value depends on real HOPE not escaping the degeneracy, and per-level momentum, Muon or M3 may act as implicit retention. A faithful HOPE reimplementation is the expensive, risky part, and the plan becomes a short note if real CMS escapes.

**One line.** In Nested-Learning/HOPE-style memories, levels trained on one shared loss at different update frequencies are, in the linear and first-order regime, exact lagged and scaled copies of the fastest level. They therefore forget on the fast timescale and add nothing across a session reset. We prove this, derive the retention and downscaling schedule that restores real consolidation along with an emergent spacing optimum, and measure how far the degeneracy survives in nonlinear CMS, DeltaNet+CMS on MQAR, and a HOPE reimplementation.

**Memory store.** Cross-store: fast state (delta-rule / fastest CMS level) to slow weight levels inside one multi-timescale memory

**Task.** Associative recall with session resets, where fast state is wiped and slow levels are kept. Three settings: (i) linear key-value streams with recurring keys at controlled gaps among random distractors (CPU); (ii) Zoology MQAR with key sets recurring within a session vs across sessions; (iii) Rotated MNIST with 20 rotations revisited on massed vs spaced schedules, using a 3-level MLP CMS. Plus a mechanism audit of a small community HOPE reimplementation that streams text with inserted synthetic facts.

**Gap in the big picture.** The doc says fast state is 'lost when the session ends unless consolidated'. It names consolidation (what moves into the weights, and when) as the open problem, and calls multi-timescale memory (Nested Learning/HOPE CMS) 'the main bet'. CMS assumes that a lower update frequency yields a slower memory, but no one has checked whether levels that share one objective actually separate timescales or transfer content. Our pilot shows that in the linear case they do not.

**Hypothesis.** Take a multi-frequency memory whose levels are trained by (chunked) SGD on one shared loss, with no per-level retention asymmetry (decay, downscaling or reset). After a fast-level reset, slow-level recall of items last seen more than tau_fast = d/(beta(2-beta)*load) steps earlier will be at most 0.35 of a slow-only learner's and will decrease monotonically with spacing. Adding fast-level retention A_f<1, or periodic downscaling, will produce an interior spacing optimum within 25% of the closed-form g*(A_f, d, load, C_k) and at least 2x post-reset retention. Both effects will persist qualitatively in DeltaNet+MLP-CMS on MQAR.

**Method.**

(1) Degeneracy lemma. Let W = sum_k W_k, where level k applies the accumulated gradient of the shared loss every C_k steps with learning rate eta_k (plain SGD, no per-level decay). Then W_k(t) = (eta_k/eta_1) * W_1(last C_k boundary), exactly, so slower levels are lagged and scaled copies of the fastest. Corollary: post-reset content is only what the fast level held at the last boundary, so spacing gives no benefit. For residual-chained nonlinear levels, x + f_1(x) + f_2(x + f_1(x)), the lemma holds to first order, and the deviation grows with the cross-term ||f_fast||*||f_slow||. That is a testable prediction for real CMS. (2) Theory under symmetry breaking. Generalise the shared-error two-state model to K levels with retention A_k, learning rate eta_k, period C_k, and delta-rule interference at load L in dimension d. Each item's coefficients obey da_k/dt = -(eta_k L/d) * sum_j a_j - (1-A_k) a_k. This gives closed-form memory curves and an invariant: a_f - (beta/eta) a_s is conserved without decay, so a single shared-residual write leaves no durable slow trace. It also gives post-reset retention, the spacing optimum g*, and the in-session cost. (3) The management rule is periodic downscaling, W_fast <- (1-gamma) W_fast every S steps, which is equivalent at first order to A_f = (1-gamma)^(1/S). We derive gamma*(S, L, d, C_k) to maximise post-reset SNR within an in-session recall budget, and pick C_k ratios to match. (4) Compare head-to-head with explicit transfer rules: Benna-Fusi coupled chain, CLS-style distillation at session end, copy-fast-into-slow before reset (lossless for parallel linear levels but it imports distractor noise, so we measure SNR), and Gated DeltaNet's learned decay. Test in linear sims, an MLP CMS, DeltaNet+2 slow MLP levels updated by the inner key-to-value loss at periods C=(16,256), and the HOPE reimplementation. For the last, the audit measures (a) the functional lag-correlation between slow-block and lagged fast-block output changes on probe inputs and (b) post-reset retention with and without the derived schedule.

**Datasets.**

- Synthetic linear key-value streams with recurring keys, controlled gaps and Gaussian/unit-norm distractors at varied load (custom, CPU)
- MQAR (multi-query associative recall) from HazyResearch/zoology (zoology/data/associative_recall.py), modified for session resets and key sets recurring massed vs spaced
- Rotated MNIST (torchvision MNIST), 20 rotations revisited on massed vs spaced schedules
- WikiText-103 (small slice) with inserted synthetic key-value facts, used only for the HOPE-reimplementation mechanism audit

**Models / checkpoints.**

- Custom NumPy/JAX linear multi-level memories (delta-rule fast level plus 1-3 slow levels; CPU)
- 3-level MLP Continuum Memory System, about 1M params, PyTorch (C = 1, 16, 256)
- fla-org/flash-linear-attention: fla.layers.DeltaNet (fast state) plus 2 slow MLP levels, 2-4 layers, <=20M params, trained from scratch on MQAR via Zoology
- fla-org/flash-linear-attention Gated DeltaNet (gated_delta_rule) as the learned-decay baseline at the same scale
- kmccleary3301/nested_learning (community mechanism-level HOPE/CMS reimplementation; hope_attention / hope_selfmod variants, pilot_smoke-scale configs; cms_fast test-time teach-signal updates)

**Baselines.**

- Single fast level (delta rule / DeltaNet state) reset at session end
- Slow-only learner (same eta, no fast level), the upper reference for slow retention
- CMS shared-residual rule with no per-level retention (HOPE-like), C = (1,16,256)
- Fast-weight decay (Hinton & Plaut 1987; Ba et al. 2016 lambda-decay)
- Gated DeltaNet learned decay gate
- Two-state shared-error model with retention A_f < A_s
- Benna-Fusi coupled-chain transfer
- CLS-style distillation of the fast level into slow levels at session end (extra pass)
- Copy-fast-into-slow before reset
- Experience replay with matched compute (data-buffer reference)

**Metrics.**

- Post-reset recall (1 - normalised MSE for linear; exact-match accuracy for MQAR) vs inter-repetition gap
- Slow-level signal coefficient and memory-curve SNR(t) (Benna-Fusi style)
- Theory fit: R^2 of predicted vs simulated retention curves; relative error of the predicted g* and gamma*
- Degeneracy index: ||W_slow - (eta_s/eta_f) W_fast(boundary)|| / ||W_slow|| (linear), and the functional lag-correlation of slow vs fast block output changes (nonlinear/HOPE)
- In-session recall cost of each retention/downscaling schedule
- Rotated-MNIST average accuracy and average forgetting after a fast-level reset
- Extra FLOPs/latency per consolidation rule

**First experiment (go/no-go).**

A 10-second CPU pilot is already done (d=128, beta=0.5, eta=0.02, C=16, 4 presentations, 32 targets, script in the session scratchpad cms_spacing_sim.py / sim2.py / lemma.py). It confirmed the lemma to machine precision: ||W_s-(eta/beta)W_f||/||W_s|| = 3e-15. With the plain shared residual, the slow-level item coefficient fell from 0.024 to 0.014 as the gap grew from 1 to 2048, against 0.077 for slow-only: no spacing benefit. With fast decay A_f=0.97 it peaked at 0.067 at gap 128 (2.8x), and downscaling with gamma=0.5 every 64 steps matched A_f=0.99. Week 1 (CPU): scale this to d in {64,128,256}, load in {0.5,1,2}, beta in {0.25,0.5,1}, C in {1,16,256}, 2-4 levels, gaps 1-4096 and 10 seeds; derive and fit closed forms for retention, g* and gamma*. Week 2 (one 24 GB GPU, <=10 GPU-h): train DeltaNet (fla) + 2 slow MLP levels, updated at test time by the inner key-to-value loss at C=(16,256), on Zoology MQAR. Evaluate cross-session recall after resetting the DeltaNet state, with key sets repeated massed vs spaced, for plain CMS vs CMS + derived downscaling vs slow-only. GO if the linear theory fits (R^2>=0.9, g* within 25%) AND, in DeltaNet+MLP, plain CMS retains <=50% of slow-only post-reset recall or shows no spacing benefit while the derived schedule gives >=1.5x.

**Kill criterion.**

Drop, or shrink to a workshop note, if any of the following holds. (1) The day-1 live novelty search finds a 2025-26 paper showing the lagged-copy degeneracy, or a shared-error multi-level consolidation analysis, for CMS/HOPE or for test-time-training memories. (2) In the nonlinear settings (MLP CMS on Rotated MNIST, DeltaNet+MLP on MQAR, HOPE-reimplementation audit), plain shared-objective slow levels already keep >=80% of slow-only post-reset recall and show a spacing benefit, which would make the degeneracy a linear artifact that real CMS escapes. (3) The derived retention/downscaling schedule improves post-reset MQAR recall by <10% relative over plain CMS at matched in-session accuracy.

**Compute.** 130 GPU-hours over 11 weeks. 1x 24-48 GB GPU (RTX 4090 / A6000 / L40S), plus an 8-16 core CPU for the linear simulations (about 20-50 CPU-hours)

**Risks.**

(1) The lemma is trivial once stated, so reviewers may call it obvious. The paper's weight must therefore rest on the nonlinear/HOPE audit and the closed-form design rule, and must cite Smith et al. 2006 and Hinton & Plaut 1987 up front. (2) Real HOPE uses chained MLPs and per-level momentum, Adam or Muon, which break exact proportionality. The degeneracy may be weak in practice (kill criterion 2), so this is the main scientific risk, and it is cheap to test in week 2. (3) The community reimplementation may deviate from the paper's CMS, so audit conclusions must be scoped to it and to our own minimal CMS. (4) Downscaling is first-order equivalent to continuous decay, so it is not a new mechanism; the novelty is the schedule and theory. (5) There is a real in-session accuracy cost, which must be reported as a Pareto curve. (6) Rotated MNIST is weak evidence and MQAR is simplistic, so a reviewer may ask for an LM-scale check that the budget cannot fully cover. (7) The novelty check remains low-confidence, and a 2026 CMS-analysis paper may exist. Optional stretch, only if time allows: the same shared-loss fast/slow analysis applies to adding a slow level beside CUT3R's state under TTT3R's update rule, which is the doc's 3D section.

**Target venue.** ICML 2027 (deadline about late Jan 2027); fallback TMLR or CoLLAs 2027, plus a NeurIPS 2027 continual-learning/memory workshop

**First-round prior-work verdict: partially_novel.** The original claim, that residual starvation yields a spacing effect and downscaling fixes it, is only partly defensible. My own pilot falsified H1 in its plain form: with a shared residual and no per-level retention, slow levels are exact lagged copies of the fast level, so spacing does not help. Fast-weight decay as a consolidation aid also has precedent (Hinton & Plaut 1987; Ba et al. 2016; Gated DeltaNet's decay gate). The shared-error fast/slow two-state model from motor adaptation (Smith, Ghazizadeh & Shadmehr 2006) is a must-cite precedent showing that timescales come from retention factors; I know it from background knowledge only and did not verify it this run. Mozer 2009, Benna-Fusi and Go-CLS cover spacing and consolidation in cognitive and synaptic models. The defensible delta has three parts: (a) the degeneracy lemma and first-order extension for CMS/HOPE-style multi-frequency levels on a shared loss; (b) a closed-form design rule for retention/downscaling and C_k under delta-rule load; (c) an empirical audit of whether real nonlinear CMS escapes the degeneracy. Confidence is LOW. The web-search budget was exhausted again this run, arxiv/nature/mlr domains failed DNS, and links for the classic papers below come from memory. Only the GitHub reimplementation, flash-linear-attention and Zoology were fetched and verified. Rerun a live search on day 1: 'Nested Learning follow-up 2026', 'continuum memory system analysis', 'two-state model test-time training', 'fast weight decay consolidation'.

**Closest work found in the first round:**

- [Nested Learning: The Illusion of Deep Learning Architectures (HOPE / Continuum Memory System)](https://arxiv.org/abs/2512.24695) — arXiv 2025 (ID 2512.24695 as given in the idea; not verified this run). Proposes the multi-frequency CMS levels updated at different periods on a shared objective, which is the exact substrate the idea analyses. The idea says it has no transfer or spacing analysis between levels. I could not confirm that because the fetch failed (DNS error), so a later version or appendix might cover it.
- Predicting the optimal spacing of study: a multiscale context model of memory (Mozer et al.) — NIPS 2009 (named in the idea; not verified this run). Derives spacing effects and optimal inter-study intervals from a model with several timescales, and the optimum depends on the retention interval. This is conceptually close to H1, which predicts an interior optimal gap between tau_fast and tau_slow. Mozer's model is a cognitive model, not gradient-coupled fast-weight levels, but it is the most likely reason a reviewer would say the effect was already known.
- Computational principles of synaptic memory consolidation (Benna & Fusi) — Nature Neuroscience 2016 (named in the idea; not verified this run). Derives closed-form memory curves and SNR(t) for chains of coupled variables with different timescales. The idea uses this as an explicit transfer baseline, and its SNR-curve analysis method is the same kind of analysis the idea proposes.
- Continual Reinforcement Learning with Complex Synapses (Kaplanis, Shanahan, Clopath) — ICML 2018 (named in the idea; not verified this run). Applies Benna-Fusi multi-timescale synapses to continual RL. Transfer happens through coupling inside each weight, not through levels that share one loss.
- Organizing memories for generalization in complementary learning systems / Go-CLS (Sun et al.) — Nature Neuroscience 2023 (named in the idea; not verified this run). Analytical teacher-student theory of when fast (hippocampal) content should be consolidated into slow (neocortical) weights. This is close to 'what moves into the weights, and when', but the trigger is predictability, not spacing or a shared residual.

### Second-round re-check (live search)

**Revised one line.** Multi-timescale learners only separate timescales through retention asymmetry (Jones et al. 2023; Smith et al. 2006). We first show that Nested-Learning/HOPE CMS levels, trained on one shared loss at different update periods, inherit the no-retention degeneracy: they are lagged copies of the fast level and carry nothing across a session reset. We then derive a load-aware downscaling and period schedule for delta-rule fast states, and test, on DeltaNet+MLP-CMS (MQAR) and a HOPE reimplementation, whether nonlinear CMS escapes the degeneracy, and how the derived schedule compares with distillation-based 'sleep' consolidation.

**Verdict: partially_novel; recommendation: pursue_with_pivot.**

**What is taken and what is still new.** I ran 12 live searches and found no paper that scoops the CMS-specific claim. Specifically, I found no paper showing that Nested-Learning/HOPE CMS levels trained on one shared loss at different update periods are lagged, scaled copies of the fastest level and so do not consolidate across a session reset. I found no independent CMS ablation or critique that separates update frequency from retention. The only substantive Nested Learning follow-up I found is from the same group: 'Language Models Need Sleep', arXiv 2606.03979, June 2026. It adds explicit consolidation (upward distillation, 'Knowledge Seeding', plus RL 'Dreaming' replay) rather than analysing whether frequency alone consolidates. Indirectly, this supports the plan's premise.

The general principle, however, is clearly prior art and must not be claimed:
(a) Jones et al., ICLR 2023 ('Learning in Temporally Structured Environments'). Its weights are sums of subweights with separate learning and decay rates on a shared error. It proves equivalences between coupled-timescale models and that multiscale learner, and shows that momentum is equivalent to a fast weight. The plan's K-level ODE and its 'no decay means no separation' invariant fall inside this framework.
(b) Smith, Ghazizadeh & Shadmehr 2006. Timescales come from retention plus error sensitivity under a shared error.
(c) Hinton & Plaut 1987. Fast and slow weights share an error and the fast weights decay, which is the plan's fix.
(d) Benna & Fusi 2016 and Mozer et al. 2009 already produce spacing effects and optimal-spacing intervals from multi-timescale memories.

What remains defensible:
(1) The discrete-period, chunk-accumulated form of the degeneracy, specific to CMS/HOPE (period C_k, not learning rate), stated as a corollary of Jones/Smith. It also gives an operational test that update frequency is not a timescale in deployed architectures.
(2) A closed-form design rule (gamma*, C_k ratios, g*) that includes delta-rule capacity and load interference in dimension d. The cognitive and synaptic models lack this.
(3) The empirical audit of whether nonlinear, residual-chained CMS (DeltaNet+MLP on MQAR, plus a HOPE reimplementation) escapes the degeneracy, measured as cross-session recall after a fast-state reset.
(4) Head-to-head against the Sleep paper's distillation-style consolidation at matched compute.

The contribution is a modern-architecture audit plus a design rule, not a new principle. Confidence is moderate: I read only snippets, and the Nested Learning appendix and the Sleep paper's full text could not be opened.

**Required revisions.** 1. Present the degeneracy lemma and the K-level shared-error ODE as corollaries or specialisations of Jones et al. ICLR 2023 (multiscale learner, sum of subweights with learning and decay rates, reparameterisation equivalences) and of Smith et al. 2006. Cite both in the intro, and drop 'we prove' language that implies a new principle. Claim only the discrete-period C_k extension, the delta-rule load term, and the session-reset metric as new.
2. Drop or soften 'emergent spacing optimum' as a headline. Benna-Fusi 2016 (explicitly covers spacing effects) and Mozer 2009 already show it. Keep only the load- and d-dependent closed form for g* as the sharper, falsifiable prediction.
3. Add 'Language Models Need Sleep' (arXiv 2606.03979) as the explicit-transfer baseline and as the key related-work contrast. Their move to explicit distillation-based consolidation is indirect evidence for the plan's thesis. Reframe the CLS-distillation baseline as a matched-compute, small-scale proxy of Knowledge Seeding.
4. Correct the closest-work list: arXiv 2512.24695 is verified (Behrouz et al., NeurIPS 2025). Benna-Fusi's arXiv preprint is 1507.07580. Add the eLife 2025 fast/slow plasticity preprint as shared-error precedent.
5. Weight the paper on kill criterion 2, the nonlinear/HOPE audit. If real CMS escapes the degeneracy, the paper becomes a short note. Before claiming the degeneracy for real HOPE, check whether its CMS uses per-level momentum, Muon or M3, or chunk-accumulated gradients that act as implicit retention, because Jones shows momentum is equivalent to a fast weight with negative learning rate. Also compare to a Gated-DeltaNet decay gate.
6. Kill criterion 1 is currently not triggered, but repeat a targeted search for papers citing 2512.24695 and 2606.03979 before submission.

**Baselines to add.**

- Language Models Need Sleep (Behrouz et al., arXiv 2606.03979): Knowledge-Seeding distillation consolidation, reimplemented at small scale as fast-into-slow distillation at session end, plus an optional replay ('Dreaming'-lite) at matched compute
- Jones et al. ICLR 2023 multiscale learner: sum of subweights with learned or fixed per-level learning and decay rates, as the principled retention-asymmetry baseline
- Momentum/optimizer-state ablation: HOPE CMS with per-level momentum or M3 vs plain SGD, to test whether optimizer state already acts as implicit retention
- Gated DeltaNet decay gate (already listed); keep it, and add a learned per-level decay on slow CMS levels

**Closest work (second round):**

- [Nested Learning: The Illusion of Deep Learning Architectures (Behrouz, Razaviyayn, Zhong, Mirrokni; HOPE / Continuum Memory System)](https://arxiv.org/abs/2512.24695v1) — NeurIPS 2025 / arXiv 2512.24695 (v1 31 Dec 2025). The ID is verified as real.. This is the substrate the plan analyses: CMS MLP blocks updated at different frequencies, with the claim that lower frequency gives more persistent memory and a 'loop' by which forgotten knowledge can partly be recovered. Search summaries describe no ablation that isolates update frequency from retention and no analysis of transfer between levels, so the plan's target gap looks open. I could not read the full text or appendix.
- [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories (Behrouz, Hashemi, Mirrokni et al.)](https://arxiv.org/pdf/2606.03979) — arXiv 2606.03979 (June 2026). The ID is verified as real.. This is a follow-up from the same group that tackles the plan's open problem directly: moving in-context, short-term memory into long-term parameters. It does so with explicit consolidation, 'Knowledge Seeding' (upward distillation from a smaller model into a larger one), plus RL-driven 'Dreaming' replay. It also formalises update frequency for CMS-like components. It does not, as far as snippets show, present the shared-loss lagged-copy degeneracy or a spacing theory. It does, however, implicitly concede that frequency alone does not consolidate, and it is now the main explicit-transfer baseline and framing threat.
- [Learning in Temporally Structured Environments (Jones et al.; workshop version 'Learning at Multiple Timescales')](https://iclr.cc/virtual/2023/poster/10732) — ICLR 2023 (NeurIPS 2022 MemARI workshop). This is the most dangerous precedent and it is missing from the plan. Each weight is a sum of subweights with different learning AND decay rates, all trained on the same error. The paper proves that coupled-timescale models reduce to this multiscale learner by reparameterisation, and that momentum equals a fast weight with a negative learning rate. The plan's K-level ODE (shared error, retention A_k) and its 'no decay means no timescale separation' degeneracy are essentially special cases of this framework. Only the discrete periods C_k and the delta-rule load are new.
- [Interacting Adaptive Processes with Different Timescales Underlie Short-Term Motor Learning (Smith, Ghazizadeh, Shadmehr)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1463025) — PLoS Biology 2006 (verified). This is the shared-error two-state model in which the processes differ by error sensitivity and retention. It predicts spontaneous recovery and savings. The plan's statement that timescales come from retention factors rather than learning rate or update frequency is this model's core principle.
- [Using fast weights to deblur old memories (Hinton & Plaut)](https://www.cnbc.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html) — CogSci 1987 (verified). Each connection has a slow and a fast weight trained on the same error, and the fast weight decays toward zero. This is exactly 'shared residual + fast-level retention A_f<1', the plan's fix.
- [Computational principles of synaptic memory consolidation (Benna & Fusi)](https://arxiv.org/abs/1507.07580v1) — Nature Neuroscience 2016 (verified; arXiv preprint 1507.07580). Coupled multi-timescale synaptic variables give closed-form memory curves and SNR(t), and the abstract states that the model explains spacing effects. This anticipates the plan's 'emergent spacing optimum' claim in a coupled-chain setting.
- [Predicting the optimal spacing of study: a multiscale context model of memory (Mozer, Pashler, Cepeda, Lindsey, Vul)](https://www.yorku.ca/ncepeda/publications/MPCLV2009.pdf) — NIPS 2009 (verified). Derives a non-monotonic spacing-retention curve and an optimal inter-study interval from multiscale traces. This is prior art for the interior spacing optimum g*.
- [Fast and slow synaptic plasticity enables concurrent control and learning](https://elifesciences.org/reviewed-preprints/105043) — eLife reviewed preprint 2025. Fast and slow weights are driven by the same feedback signal. The fast weights correct immediate errors and the slow weights gradually absorb what the fast ones were correcting. This is again shared-error fast/slow consolidation, in a control setting, with no spacing or session-reset analysis.
- [Organizing memories for generalization in complementary learning systems / Go-CLS (Sun, Advani, Spruston, Saxe, Fitzgerald)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400413) — Nature Neuroscience 2023 (verified). Analytical theory of when fast-store content should be consolidated into slow weights: only when doing so improves generalization. This is relevant to 'what moves into weights, and when', but the trigger is predictability, not spacing or retention asymmetry.

**Queries run:** Nested Learning continuum memory system analysis frequency levels timescale arXiv 2026 (extended); arXiv 2512.24695 Nested Learning Illusion of Deep Learning Architectures; "Language Models Need Sleep" Learning to Self-Modify and Consolidate Memories; continuum memory system HOPE slow levels lagged copy fast level same objective degenerate analysis; multi-timescale fast and slow weights shared error consolidation spacing effect theory neural network 2025 (extended); Smith Ghazizadeh Shadmehr 2006 interacting adaptive processes different timescales motor learning two-state model; consolidating fast weights into slow weights test-time training linear attention state reset session 2026 arXiv; spacing effect neural network optimal spacing gap memory model 2025 arXiv spaced repetition linear associative memory; critique OR analysis of Nested Learning HOPE continuum memory system update frequency ablation forgetting 2026; Hinton Plaut 1987 "Using fast weights to deblur old memories"; Matt Jones "Learning at Multiple Timescales" subweights decay sum of weights equivalent single learning rate; Benna Fusi 2016 computational principles of synaptic memory consolidation; Sun 2023 organizing memories for generalization complementary learning systems; theory of multi-frequency test-time memory levels update period chunk size linear regression Titans HOPE when do slow levels help 2026 arXiv

---

## 6. Displacement-Aware Experience Memory: Margin-Gated Contrastive Key Rewriting Against Retrieval-Side Forgetting in Sequential LLM Agents

*Added in the second round for an angle the first round missed.*

**Judge:** overall 6/10, tier B. Scores out of 5: novelty 3, low compute 5, impact 3, speed of first signal 4.
This is the cheapest plan in the set: about 30 GPU-hours plus $50-150 of API calls. It turns the doc's lesson that external memory moves the dilemma (2604.27003) into a measurable mechanism, an oracle that removes only the newer same-group competitors. It also tests the Crowded Embedding Space prediction inside a real agent loop. The oracle-first gate keeps sunk cost small.

**Biggest weakness:** 2603.02473 shows write strategy matters far less than the retrieval method, so effects may be smaller than seed noise for a gpt-4o-mini ALFWorld agent. The 2604.27003 protocol may not reproduce, and REALM may already contain contrastive restructuring. Every component besides the displacement measurement has close prior art.

**One line.** This plan is a training-free study of how newer experiences push older ones out of a capped retrieval budget in sequential agent memory (ALFWorld/BabyAI, protocol of arXiv 2604.27003). It tests whether margin-gated contrastive key rewriting plus a one-slot-per-competition-group cap can reduce forgetting (BWT) without losing forward transfer. Plain "self-relevance" key optimisation serves as a falsifiable foil: the Crowded Embedding Space theory (ICML 2026) predicts it will make crowding worse. The conversational LongMemEval part is now secondary, and its supersession component is dropped as prior art.

**Memory store.** External store: the write head (key construction, plus the choice between rewriting and appending) and the read head (which retrieved entries get context slots, and how many). The output is a per-group collision log. That log feeds the big-picture doc's open consolidation problem: groups that keep colliding are the candidates for promotion into weights or for summarisation into one procedure. The 3D link (Point3R/TTT3R) is kept only as a stated analogy and is out of budget.

**Task.** Primary task: reusing experience in sequential agent tasks under a capped context. The setup follows 2604.27003: ALFWorld (6 task types as a sequential stream) and BabyAI. Memory is an external (k,v) store of abstract procedures plus trajectories, read by a frozen API LLM agent. Secondary task: long-term conversational memory QA under a fixed token budget, run only through existing collision and interference protocols (Entity-Collision-style collision degrees on LongMemEval_S, and a LongMemEval_M natural-density slice), not a new benchmark.

**Gap in the big picture.** 2604.27003 (https://www.emergentmind.com/papers/2604.27003) shows that capped-context external memory moves continual-learning difficulty into retrieval. Old and new experiences compete there, and designs with strong forward transfer can forget badly. The paper diagnoses this but proposes no management method, and no measurement of how much forgetting comes from slot displacement.

Each piece of the original De-Shadow plan is already covered elsewhere:
- Stress-testing with confusable distractors: MINTEval (2605.18565), Entity-Collision (2605.29630), the personal-memory conflict paper (2608.13921) and MEME.
- An analytic crowding model: Crowded Embedding Space (2606.28343, ICML 2026), AdaWidth (2608.23862) and the order-statistics view in Curse of Dense Low-Dim IR (2012.14210).
- Write-time supersession and version ledgers: MemStrata (2606.26511) and MoM (2609.25054).
- Retrieval-feedback reconsolidation: REALM (2609.16053).
- Diversity-aware, redundancy-reducing retrieval: xMemory (2602.02007).

Still uncovered:
- (a) Measuring directional displacement, where new entries crowd out old ones, as the mechanism behind BWT in experience memory for sequential agent tasks.
- (b) A write-time key-space intervention, contrastive rewriting of confusable keys accepted only if a label-free probe margin improves, aimed at that forgetting.
- (c) A test of the Crowded Embedding Space prediction that locally self-relevant key optimisation marginalises minority (older or rarer) task types, against a contrastive alternative, inside a real agent memory loop.

**Hypothesis.** H1 (mechanism): On the 2604.27003 ALFWorld/BabyAI stream with a capped context, most backward-transfer loss on earlier task types comes from displacement. Displacement means newer, same-neighbourhood procedure entries outrank the entry that helped the earlier task and take its slots. The test is an oracle that removes only those newer competitors from old-task queries. It should recover at least 50% of the BWT gap at equal tokens.

H2 (foil): Self-relevance key optimisation means each key is rewritten so its own probe queries retrieve it, with A-Mem-style key evolution as the example. Following Crowded Embedding Space, it should raise displacement of minority or older task types and worsen BWT, even if mean recall rises.

H3 (intervention): Two training-free changes should improve BWT by at least 3 success-rate points with at most a 1-point loss in FWT, compared with the best 2604.27003 configuration and with xMemory/MMR-style diverse retrieval at equal context tokens:
- Contrastive key rewriting: an LLM adds distinguishing attributes to both keys of a confusable pair, and the rewrite is accepted only if the pairwise probe margin rises for both entries.
- A one-slot-per-collision-group cap with adaptive k.

**Method.**

0) Instrumentation, CPU only.
- Reproduce 2604.27003's (k,v) configurations: abstract procedure vs trajectory values, coarse vs fine keys.
- For every retrieval, log the candidate ranks and the task-type and timestamp of each entry.
- Define a utility-labelled "helpful entry" per old-task query: the entry whose single-slot inclusion maximises success, estimated on a small calibration subset.
- Displacement Rate: the fraction of old-task queries where a newer entry from the same collision group outranks the helpful entry out of the budget.
- Use the existing order-statistics and mean-field results as a predictive tool, not as a contribution. Fit Delta/sigma per embedder and check whether displacement follows the predicted dependence on neighbourhood density.

1) Margin-gated contrastive key rewriting (write path, training-free).
- On writing entry e, find competitors with cosine above tau_c. tau_c is the 99th percentile of random-pair similarity for Qwen3-Embedding-0.6B.
- gpt-4o-mini, pinned and at temperature 0, generates 2 probe queries per entry from the value, not the key.
- For each confusable pair (e, j), the LLM rewrites both keys to add distinguishing attributes: task type, object, room, precondition, outcome.
- Accept the rewrite only if, for both e and j, own-probe score minus cross-probe score increases and no other group member's margin drops. This is the contrastive objective.
- Values are never rewritten, so there is no rewrite drift (the failure documented in 2605.12978).

2) Foil, self-relevance rewriting. The same LLM and budget rewrite each key to maximise its own probe score only. The acceptance rule is identical except the cross-entry term is dropped.

3) Read path.
- Hybrid BM25 plus dense top-50.
- Group candidates by the collision links logged at write time. Each group gets at most one slot, filled by its highest-scoring member.
- Adaptive k*: stop at a large score gap or when the token budget runs out.

4) Ablations: read cap only, contrastive rewrite only, both, self-relevance foil, and oracle de-displacement. Report write-time tokens per entry.

5) Secondary conversational check. Apply the same write and read path to LongMemEval_S under Entity-Collision-style collision degrees, plus a LongMemEval_M slice. Report Displacement Rate and QA. MemStrata-style supersession is a baseline, not our contribution.

**Datasets.**

- ALFWorld (6 household task types as a sequential task stream), protocol of arXiv 2604.27003
- BabyAI (sequential levels), protocol of arXiv 2604.27003
- LongMemEval_S and LongMemEval_M cleaned (Wu et al., ICLR 2025), secondary only
- Entity-Collision protocol (arXiv 2605.29630) applied to LongMemEval_S, for controlled collision degree instead of a new injected benchmark

**Models / checkpoints.**

- Qwen/Qwen3-Embedding-0.6B (frozen main embedder)
- BAAI/bge-m3 (second embedder, to check the result holds across embedders)
- sentence-transformers/all-MiniLM-L6-v2 (cheap embedder; the best overall in Entity-Collision)
- Qwen/Qwen3-Reranker-0.6B (optional)
- BM25 plus hybrid retriever
- gpt-4o-mini or the cheapest pinned API snapshot (agent, probe generator, key rewriter); match 2604.27003's backbone if it is API-accessible
- GPT-4o with LongMemEval's official judge prompt (secondary QA only)

**Baselines.**

- 2604.27003 (k,v) configurations: detailed trajectories vs abstract procedures; coarse vs fine keys
- No-memory agent and append-only flat top-k memory
- MMR diversity re-ranking (lambda in {0.3, 0.5, 0.7}) at equal tokens
- xMemory-style hierarchical, diversity-aware retrieval (arXiv 2602.02007)
- A-Mem-style key/note evolution, equivalent to the self-relevance foil (Zettelkasten-style write-time linking)
- Mem0 ADD/UPDATE/DELETE with the same embedder and reader
- MemStrata-style deterministic supersession (arXiv 2606.26511), on the conversational track
- REALM retrieval-feedback reconsolidation (arXiv 2609.16053), if code is released; otherwise a re-implementation of its co-utilisation clustering step
- LongMemEval fact-augmented key expansion, on the conversational track
- Recency-weighted scoring (cos + lambda*recency)
- Oracle de-displacement (newer same-group competitors removed for old-task queries), as an upper bound

**Metrics.**

- Agent success rate per task type, plus the full sequential task matrix: FWT and BWT/forgetting, as in 2604.27003
- Displacement Rate (new over old and old over new) and the share of BWT gap recovered by the oracle
- Share of oracle headroom recovered by each method
- Probe margin (own minus cross) and fitted Delta/sigma before and after rewriting; predicted vs observed displacement-vs-density fit
- Minority marginalisation: recall of the rarest or oldest task types under the self-relevance foil vs contrastive rewriting
- Write cost: LLM tokens and dollars per stored entry
- Secondary: LongMemEval QA by type (official judge, with a 100-item human check), Recall_all@k, NDCG@k
- Paired bootstrap 95% CIs over episodes and questions

**First experiment (go/no-go).**

Go/no-go in 2 weeks, using about 3 GPU-hours for embeddings and $50-150 of API calls.

Week 1:
- Reproduce the 2604.27003 ALFWorld 6-task-type stream with the abstract-procedure memory and gpt-4o-mini, at a capped budget of k=3 entries.
- Log the full candidate rankings and compute the sequential matrix (FWT, BWT).
- Run oracle de-displacement: for old-task evaluations, drop newer entries in the same collision group from the candidate pool.
- Measure the Displacement Rate and how much of the BWT gap the oracle recovers.

Week 2:
- Implement contrastive key rewriting with the margin gate, the self-relevance foil, and the per-group slot cap.
- Run each, plus MMR and recency weighting, on the same stream with 3 seeds and the same token budget.

GO if all four hold:
1. The oracle recovers at least 50% of the BWT gap, or at least 4 success-rate points.
2. The full method recovers at least 40% of the oracle headroom.
3. The method beats the best baseline on BWT by at least 3 points, with at most a 1-point FWT loss (paired-bootstrap CI excludes 0).
4. The foil shows the predicted higher displacement of minority or old task types. This is the result that makes the paper interesting even if the method is only moderate.

**Kill criterion.**

Drop the method paper if any of these holds:
(i) The oracle de-displacement recovers less than 3 success-rate points of BWT. Forgetting would then be driven by reader or negative transfer rather than slot competition.
(ii) Displacement explains less than 25% of old-task failures.
(iii) Contrastive rewriting plus the slot cap recovers less than 20% of the oracle headroom, or is matched within 1 point by MMR or xMemory-style diverse retrieval.
(iv) The gpt-4o-mini agent shows no measurable forgetting on the stream, even with the stronger backbone used in 2604.27003.

If only the foil effect, self-relevance marginalisation, replicates, downgrade to a short empirical note testing the Crowded Embedding Space prediction in agent memory.

**Compute.** 30 GPU-hours over 8 weeks.

**Risks.**

1) Heavy prior art on the pieces. Benchmark-style distractor injection (MINTEval, Entity-Collision, 2608.13921, MEME), the analytic crowding model (Crowded Embedding Space, AdaWidth), supersession and version ledgers (MemStrata, MoM), and retrieval-feedback reconsolidation (REALM) are all taken. The paper must claim only three things:
- the directional displacement measurement of forgetting in experience memory;
- the margin-gated contrastive rewrite;
- the empirical test of the self-relevance foil.

2) A 2026 study reports that write strategy matters much less than retrieval method on LoCoMo (arXiv 2603.02473). Key rewriting may therefore have small effects. The oracle-first go/no-go guards against this.

3) The gpt-4o-mini ALFWorld agent may be too weak or noisy to expose forgetting, and the variance across seeds may swamp 3-point effects. Mitigations: 3+ seeds, paired evaluation on identical games, and a stronger backbone if needed (higher API cost).

4) REALM's full method was not readable and may already include contrastive restructuring. Check it in week 1.

5) The LLM rewriter adds write cost proportional to the number of colliding pairs. Mitigations: only score pairs above tau_c, cache every call, and report tokens per entry.

6) Labelling the helpful entry by utility needs extra rollouts. Limit this to a small calibration subset.

7) API deprecation. Pin snapshots and cache all calls.

8) The 3D/TTT3R connection is only an analogy and is not evaluated within the budget.

**Prior-work verdict: partially_novel; recommendation: pursue_with_pivot.** Searching found no paper that measures retrieval-slot displacement as the cause of BWT on the ALFWorld/BabyAI experience-reuse stream, or that applies pairwise-contrastive, margin-gated key rewriting to agent memory. Caveats:
- I could not read REALM's full method, and it could contain a contrastive restructuring step.
- The Crowded Embedding Space dynamic analysis is close in spirit. The plan must cite it as the theory being tested, not claim the mechanism.
- Strongest risk: write strategy has been shown to matter much less than retrieval (2603.02473), so effects may be small. The oracle-first go/no-go is essential.

**What is taken and what is still new.** The original plan's main claimed contributions are each already covered by 2026 work:
- the shadowing measurement via injected confusable sessions (Entity-Collision, MINTEval, 2608.13921);
- the closed-form order-statistics model (Crowded Embedding Space at ICML 2026, AdaWidth);
- version-chain supersession (MemStrata, MoM);
- retrieval-collision-driven reorganisation (REALM);
- group-diverse slot selection (xMemory, MMR).

What remains defensible:
(1) Treating forgetting in sequential experience memory, the open problem posed by 2604.27003, as directional slot displacement. It can be measured with an oracle that removes only newer same-group competitors.
(2) A contrastive, margin-gated key rewrite. It rewrites both members of a confusable pair and accepts the change only if both own-versus-cross probe margins rise. Values are never touched.
(3) A direct test inside an agent loop of the Crowded Embedding Space prediction that locally self-relevant key optimisation (A-Mem-style evolution) marginalises minority or older content, which the contrastive objective should avoid.

The conversational supersession track is demoted to a baseline comparison.

**Closest work:**

- [When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents (arXiv 2604.27003)](https://www.emergentmind.com/papers/2604.27003) — arXiv 2026. Defines the problem: old and new experiences compete at retrieval under a capped context, and the FWT vs forgetting trade-off on ALFWorld/BabyAI. No management method and no displacement metric.
- [The Crowded Embedding Space: A Mean-Field Mechanism for Emergent Marginalization in Retrieval-Augmented Agents](https://arxiv.org/pdf/2606.28343) — ICML 2026. Analytic crowding and phase transition in top-k as majority density grows. It also analyses agents that update document embeddings to maximise retrieval and finds this marginalises minority content. This scoops the plan's analytic model and motivates the self-relevance foil.
- [Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge (MemStrata)](https://arxiv.org/pdf/2606.26511) — arXiv 2026. Deterministic write-time supersession in a bi-temporal ledger. Shows cosine cannot separate contradiction from duplication (AUROC 0.59). This scoops the version-chain merge and old-shadows-new fix.
- [MoM: Memory of Memory](https://arxiv.org/pdf/2609.25054) — arXiv 2026. Soft keys, write-time supersession and first-class conflict state for LLM-written memory.
- [Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents (REALM)](https://arxiv.org/pdf/2609.16053) — arXiv 2026. Post-retrieval reconsolidation of graph topology driven by co-utilisation (LoCoMo 75.97, LongMemEval 65.11). Overlaps with using retrieval collisions as a reorganisation signal.
- [Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory](https://arxiv.org/pdf/2605.29630) — arXiv 2026. Controlled distractors sharing the gold entity, with graded collision degrees, on agent memory and LongMemEval. This scoops the LME-I-style injected stress test.
- [MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems](https://arxiv.org/pdf/2605.18565) — arXiv 2026. Benchmark of interference from frequently updated facts, with distractor sweeps showing RAG degradation.
- [Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation (xMemory)](https://arxiv.org/html/2602.02007) — arXiv 2026. Argues top-k returns redundant correlated spans and selects a compact, diverse set top-down. Overlaps with the group slot cap and adaptive k.
- [AdaWidth: Query-Adaptive Embedding Width for Dense Retrieval](https://arxiv.org/pdf/2608.23862) — arXiv 2026. Order-statistic characterisation of gold dropping out of top-k via a count of overtaking competitors.
- [When Personal Memory Has No Single Answer: Evaluating LLM Agents under Irreducible Conflict](https://arxiv.org/pdf/2608.13921) — arXiv 2026. Sweeps same-domain distractor memories from 0 to 6 and measures degradation.
- [Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory](https://arxiv.org/html/2603.02473v1) — ICLR 2026. Finds write strategy moves accuracy by 3-8 points versus 20 points for retrieval method. This is a risk to any write-time key method.

**Queries run:** REALM retrieval-driven memory reconsolidation long-term LLM agents; LLM agent memory near-duplicate memories interference retrieval stale entries crowd out correct memory context budget (extended); memory interference benchmark LLM agents distractor memories similar entities knowledge update 2026; "interference" memory write agent rewrite keys to be distinguishable confusable memories retrieval discriminative indexing LLM; Hu Long Wang experience reuse forgetting forward transfer ALFWorld BabyAI memory retrieval competition 2604.27003; order statistics model retrieval top-k distractors similarity margin theory dense retrieval failure probability number of distractors; agent memory retrieval cluster near-duplicates one representative per cluster top-k redundancy LongMemEval LoCoMo adaptive number of memories; "Temporal Validity in Retrieval Memory" supersession stale-fact errors agents; "The Crowded Embedding Space" mean-field marginalization retrieval-augmented agents memory; "Entity-Collision" stratified protocol attributing retrieval lift agent memory

---

## 7. Where Should a Frozen CUT3R Keep a Scene It Will Revisit? Budget-Matched State, Snapshot, Replay and Adapter Memory, With a Signal for When an Adapter Is Worth It

*Added in the second round for an angle the first round missed.*

**Judge:** overall 6/10, tier B. Scores out of 5: novelty 3, low compute 2, impact 4, speed of first signal 3.
It is the most complete four-store instantiation of the doc's consolidation question in 3D, asking when a scene has earned weights, and the cross-session protocol appears unclaimed. It should not run as a standalone project. Its S1-S3 arms are P1's mechanisms, so it is best run as phase 2 of P1 on the same 7-Scenes revisit pairs, once P1 shows the state is worth keeping.

**Biggest weakness:** Every mechanism is prior art (Free Geometry, LoRA3D, ACE-G), so this is a benchmark paper. BPTT through CUT3R at 512 resolution makes the 160 GPU-hour estimate optimistic. Free Geometry reports gains of only about 3-4%, so the adapter-over-snapshot effect may be within noise, and an ACE-style head may own registration.

**One line.** This is a revisit benchmark and analysis, not a new consolidation method. A frozen CUT3R/TTT3R streams visit A, then visit B of the same scene. What A taught it is kept in one of five ways at a matched ~1-2 MB and <=60 s budget: carried recurrent state, a retrieved state snapshot, replayed keyframes, a per-scene LoRA distilled with the published Free-Geometry recipe (full-view teacher to partial-view student), or an ACE-style scene-coordinate head. The paper reports which wins as A grows, and whether a forgetting score computed without ground truth at the end of A predicts when the adapter is worth storing.

**Memory store.** Consolidation from fast state (the CUT3R/TTT3R recurrent state) into weights (a per-scene adapter, or a per-scene scene-coordinate head). Both are compared at matched bytes and compute against the external store (a retrievable state snapshot) and the data buffer (replayed keyframes). This is the 3D section of the big-picture doc, and it covers all four stores. The open question it tests is when content should move from context or store into weights, and across timescales: from within a visit to across visits.

**Task.** Streaming 3D reconstruction with cross-session revisits. A frozen CUT3R (TTT3R update rule) streams visit A of an indoor scene. Optionally, G other scenes stream in between as interference. Then it streams visit B of the same scene and outputs per-frame poses and pointmaps. Two things are scored: B's own quality (B-self), and whether B is registered into A's frame (A-anchored, which is essentially relocalization plus mapping).

**Gap in the big picture.** The adversarial search refuted three of the plan's original framings:
(a) Distilling a model's full-context predictions into a LoRA so that a reduced-context pass reproduces them is already published as test-time adaptation. Free Geometry (arXiv 2604.14048, Apr 2026) masks frames and makes partial-view features match full-view features through LoRA on VGGT and Depth Anything 3, taking under 2 minutes.
(b) Per-scene self-calibrated LoRA adapters for 3D foundation models exist. LoRA3D (ICLR 2025) builds 18 MB adapters in 5 minutes from confidence-calibrated pseudo-labels.
(c) Storing a scene in weights so a later visit can relocalize against it is classic scene-coordinate regression. ACE-G (ICCV 2025) trains a per-scene map code of a few MB in minutes on top of a generic pretrained transformer.
Light-parameter TTA for long sequences also exists: VGGT-Align uses LayerNorm-only updates.

What survives is narrower. No search result evaluates a streaming reconstruction foundation model (CUT3R/TTT3R/VGGT family) across separate sessions of the same scene. The extended search found no work that saves and reloads the state or fast weights across sessions. Every CUT3R follow-up still writes only into the within-stream state: TTT3R, TTSA3R, Rethinking the State Update Gate, ReCal3R, Info3R, FILT3R, SSR. Within-stream loop and revisit handling also stays within a single run (Ray-Aware Pointer Memory 2605.05749, Scal3R with place recognition, ABot-Recon's optional loop closure). The only multi-session reconstruction found is offline SfM (2602.20584).

Nobody compares the four stores (state carry, snapshot, keyframe buffer, weights) at matched bytes and compute for revisits. Nobody gives a ground-truth-free rule for when a scene has earned weights. The closest such rules are LLM-side surprise or forgetting gates such as SuRe, which have not been tested on a geometric fast state.

There is evidence that the regime exists. ABot-Recon (2608.27529) finds local context suffices outdoors, but says persistent context matters in small, frequently revisited indoor spaces. MERG3R and KV-Tracker report that CUT3R and TTT3R degrade on long 7-Scenes streams.

**Hypothesis.** H1 (capacity crossover, dose-response). The B-self gain of the best weights store (Free-Geometry-style LoRA on A plus K anchor frames) over the best budget-matched non-weight store is about zero at L=100 and grows with L. At L=1000 on 7-Scenes it is at least 5% relative ATE or Accuracy with a CI excluding 0. This is a deliberately lower bar than the original 15%, because Free Geometry reports only about 3-4% average gains from comparable adaptation.

H2 (earned signal). The re-query forgetting score F, computed at the end of A, gives Spearman rho >= 0.5 with the per-cell adapter-minus-snapshot gain. F re-feeds early keyframes of A against the final state with writes disabled and measures their Sim(3)-aligned disagreement with their original predictions. A frozen threshold policy fit on 7-Scenes has at most 25% of the regret of always-adapt and never-adapt on 12-Scenes and TUM. F also beats two simple predictors: L alone, and TTT3R's mean closed-form learning rate.

H3 (memory, not adaptation). The same-scene adapter built on A beats all of: Free-Geometry TTA run on B itself at equal seconds, an other-scene adapter, and LoRA3D-style pseudo-label calibration on B.

H4 (registration). For A-anchored ATE, adapter plus anchors, or adapter plus snapshot, is competitive with an ACE-style head trained on A using CUT3R's own poses as pseudo-ground truth at equal bytes. Otherwise registration memory belongs in an SCR head, and the adapter's contribution is geometry only, which is still reportable.

H5 (bounded harm). Loading a wrong-scene adapter costs at most 5% relative ATE.

**Method.**

Backbone. CUT3R cut3r_512_dpt_4_64.pth, frozen. The TTT3R training-free update rule is the default writer. Plain CUT3R is an ablation. Info3R, TTSA3R or FILT3R join as stronger writers if code exists.

Revisit protocol. A is one sequence and B a different sequence of the same scene, in a shared ground-truth frame. A's length L is in {100, 300, 1000}, set by stride. B is 200 frames. The gap G is in {0, 1, 3} interfering scenes.

Stores, all decided at the end of A, each <=~2 MB and <=60 s on one GPU:
- S1 state carry.
- S2 fp16 snapshot of the final state, retrieved by mean encoder feature.
- S3 K=8 keyframes, chosen without ground truth (confidence x coverage) and replayed into a reset state.
- S4 per-scene adapter. A rank-4 LoRA on q/v of the decoder plus the pose head, or a LayerNorm/bias-only variant (VGGT-Align-style). It is trained with the Free-Geometry objective ported to a recurrent model: the teacher is the full-A two-pass TTT3R output, and the student runs on random 4-16-frame sub-clips from a reset state. Pointmap/pose L1 is confidence-weighted. The contribution claimed is NOT this recipe, only its use as a cross-session store.
- S5 = S4 + S2.
- S6 ACE-style scene-coordinate head on frozen CUT3R encoder features. It is trained on A with CUT3R-predicted poses as pseudo-labels, at the matched 60 s budget and at its native few-minute budget. At B it relocalizes the first frames and anchors the stream. This is the strongest prior form of 'scene stored in weights for revisits'.

Earned signal. At the end of A, compute three things: F (re-query forgetting); an F-variant on the adapter's training loss after 20 steps (cheap 'distillability'); and the TTT3R learning-rate statistic. The policy: store the adapter iff F > tau, else store a snapshot. Report regret against the per-cell oracle.

Optional CPU toy: a delta-rule fast state of width d against LoRA consolidation as n/d grows, to predict where the crossover should sit.

Controls:
- Other-scene adapter.
- Equal-seconds TTA on B with the Free Geometry objective and with Test3R-style consistency.
- Single-pass-teacher adapter.
- Always-adapt, never-adapt, and oracle policies.

3 seeds and paired bootstrap 95% CIs over scenes.

**Datasets.**

- 7-Scenes (multiple sequences per scene in one ground-truth frame per scene; main go/no-go, used for long-sequence evaluation by TTT3R, Info3R, KV-Tracker, MERG3R)
- 12-Scenes (training/test sequences per room in a shared frame; held-out policy test; not verified in this session's searches)
- TUM RGB-D (fr1_desk to fr1_desk2, fr1_desk to fr1_room, and first half to second half of fr3_long_office_household; dynamics sequences as a stress test; check whether the mocap frame is shared before reporting A-anchored metrics)
- ScanNet v2 rescans (_00 to _01, about 20 rooms; changed-scene staleness; B-self metrics only; optional, first to be dropped if compute is short)

**Models / checkpoints.**

- CUT3R cut3r_512_dpt_4_64.pth (frozen, under 1B params)
- TTT3R training-free state-update rule on the same checkpoint (https://arxiv.org/pdf/2509.26645)
- Per-scene trainable parts only: a rank-4 LoRA or LayerNorm/bias adapter (about 0.1-0.5M params), and a small ACE-style scene-coordinate MLP head on frozen CUT3R encoder features (a few MB)
- Optional stronger state writers if code is public: Info3R (https://arxiv.org/pdf/2609.21938), TTSA3R (https://arxiv.org/pdf/2601.22615), ReCal3R (https://arxiv.org/pdf/2607.05356)

**Baselines.**

- CUT3R with state reset at B, and CUT3R state carry
- TTT3R state carry (main fast-state baseline)
- TTT3R final-state snapshot plus retrieval (external store, budget-matched)
- K-keyframe replay into a reset state (data buffer, budget-matched)
- Free Geometry test-time LoRA on B at equal seconds (https://arxiv.org/pdf/2604.14048; code hiteacherIamhumble/Free-Geometry, ported to CUT3R)
- Test3R-style consistency TTA on B at equal seconds
- LoRA3D-style confidence-calibrated pseudo-label LoRA on B (https://arxiv.org/pdf/2412.07746), as a per-scene self-calibration baseline
- ACE/ACE-G-style per-scene scene-coordinate head trained on A with CUT3R pseudo-poses (https://arxiv.org/pdf/2510.11605), the strongest 'scene stored in weights' relocalization baseline
- Stronger state writers (Info3R/TTSA3R/ReCal3R/FILT3R, whichever has code) as the no-weights ceiling
- Controls: other-scene adapter, single-pass-teacher adapter, always-adapt, never-adapt, and the per-cell oracle

**Metrics.**

- B-self ATE RMSE (Sim(3) Umeyama), RPE translation and rotation
- A-anchored ATE on B (Sim(3) fit on A only, applied to B), i.e. map persistence and relocalization
- Accuracy, Completion and Normal Consistency on B (7-Scenes protocol)
- Spearman rho between F and the per-cell gain; policy regret against the oracle; comparison with predictors L-only and the TTT3R learning-rate statistic
- Gain-versus-L dose-response slope with bootstrap CI
- Bytes stored, consolidation seconds, revisit FPS, peak GPU memory
- Wrong-route harm: relative ATE when another scene's adapter, snapshot or head is loaded

**First experiment (go/no-go).**

Two weeks on one GPU, 7-Scenes only, about 18 GPU-hours.

Week 1: run CUT3R and TTT3R inference and build 14 revisit pairs (2 per scene, B = 200 frames), with L in {100, 300, 1000} and G = 0. Implement:
- S1, S2, S3.
- S4 (the Free-Geometry-style LoRA, ported, plus 8 anchors), in two variants: LoRA and LayerNorm-only.
- S6, the ACE-style head on cached CUT3R encoder features, trained on A with CUT3R pseudo-poses for 60 s.
- Two equal-seconds controls: Free Geometry TTA on B, and the other-scene adapter.
Check that adapter BPTT fits in 24-48 GB in bf16. If it does not, fall back to 4-frame clips or LayerNorm-only.

Week 2: run all cells x 3 seeds. Report B-self ATE, A-anchored ATE, Acc/Comp, gain versus L, and Spearman(F, gain).

GO requires all three:
- (a) At L = 1000, the best S4/S5 beats the best of S1/S2/S3 by at least 5% relative B-self ATE or Accuracy, with a CI excluding 0.
- (b) The gain at L = 1000 is larger than at L = 100.
- (c) It beats both Free-Geometry TTA on B and the other-scene adapter.
Record separately whether S6 dominates A-anchored ATE. That decides whether the registration claim is kept.

**Kill criterion.**

Stop the weights line on 7-Scenes if any of these holds:
- (1) At L = 1000, no adapter variant beats the best budget-matched non-weight store by at least 5% relative B-self ATE or Accuracy with a CI excluding 0.
- (2) There is no rise in gain from L = 100 to L = 1000.
- (3) Free-Geometry TTA on B at equal seconds, or the other-scene adapter, matches the same-scene adapter within CIs. In that case the gain is adaptation, already published, not memory.

Partial kill: if the ACE-style head on A matches or beats adapter plus anchors on A-anchored ATE at equal bytes, drop the registration claim and report geometry only.

Secondary kill after week 4: Spearman(F, gain) < 0.3, or F no better than L alone. In either case, publish only the budget-matched store comparison as a short benchmark or negative-result note.

**Compute.** 160 GPU-hours over 10 weeks.

**Risks.**

(1) Thin novelty. The consolidation mechanism (Free Geometry), per-scene adapters (LoRA3D) and per-scene weights for revisits (ACE/ACE-G) are all prior art. The contribution is only:
- (a) the cross-session revisit protocol for streaming reconstruction foundation models;
- (b) the budget-matched comparison of stores, including SCR;
- (c) the ground-truth-free 'earned' signal and policy.
Reviewers may call it a benchmark paper, so it should be framed as one, with the memory-management framing as the lens.

(2) Small effects. Free Geometry's average gains are about 3-4%, and CUT3R on 200-frame B may already be good. That is why the dose-response at L = 1000 and the A-anchored metric are the sensitive tests.

(3) A concurrent cross-session paper is plausible. The CUT3R-writer space is crowded: TTT3R, TTSA3R, the update-gate rethink, ReCal3R, Info3R and FILT3R all appeared within about 12 months. Stronger writers or local-context models like ABot-Recon may also erase the overflow regime.

(4) Domain-adaptation confound, because 7-Scenes uses one shared Kinect. The other-scene adapter and TTA-on-B controls address this.

(5) Teacher drift is largest where F is high. Mitigations: confidence weighting, the two-pass teacher, and teacher-vs-ground-truth logged as a diagnostic.

(6) CUT3R anchors the world frame to the first frame, so an adapter alone cannot carry registration. Hence the anchors, S5 and S6.

(7) BPTT memory at 512 resolution. Fall back to short clips or LayerNorm-only.

(8) Licensing and data. CUT3R uses a custom license. ScanNet requires its terms of use. TUM frame sharing is unverified. 12-Scenes and ScanNet rescans were not verified in searches.

(9) Search-budget breach. This check ran 12 searches against a cap of 10, and paper details come from snippets and abstracts only.

**Prior-work verdict: partially_novel; recommendation: pursue_with_pivot.** 12 searches were run, which exceeds the assigned cap of 10 by 2. This should be flagged to the orchestrator. One was extended.

Three searches refuted the mechanism-level novelty:
- Free Geometry (Apr 2026) already does test-time LoRA distillation from the model's longer-context self.
- LoRA3D already stores per-scene adapters.
- ACE-G already stores scenes in per-scene weights for later relocalization.
The original plan cited none of the three.

The extended cross-session query, the multi-session query and the state-bank query all came back without a streaming foundation-model method that persists memory across sessions. The continual-TTA and decision-policy queries found no adapter library or earned-signal rule for 3D reconstruction. The surviving claim is therefore a benchmark-plus-policy paper.

Confidence is moderate. Evidence is from snippets and abstracts only. The CUT3R follow-up space moves at about one paper a month, so a concurrent 'revisit' paper remains likely. Free Geometry's code should be checked early, both for how easily it ports to CUT3R and to confirm it does not already evaluate cross-sequence reuse.

**What is taken and what is still new.** The plan's original method novelty is gone.
- Full-context to partial-context LoRA self-distillation is Free Geometry (2604.14048).
- Per-scene self-calibrated LoRA is LoRA3D.
- Per-scene weights reused for revisits and relocalization is ACE/ACE-G.

What remains defensible:
- (1) A cross-session revisit protocol for streaming reconstruction foundation models. No search result evaluates CUT3R/TTT3R/VGGT-family models across separate sessions of the same scene, or saves and reloads their state or weights.
- (2) A budget-matched comparison of all four memory stores, now including an SCR head as the classic weights store, as a function of first-visit length. This answers when state capacity runs out.
- (3) A ground-truth-free 'earned' signal and threshold policy for consolidating a fast geometric state into weights, tested for transfer across datasets.
Each prior work is used as a component or baseline, not claimed.

**Closest work:**

- [Free Geometry: Refining 3D Reconstruction from Longer Versions of Itself](https://arxiv.org/pdf/2604.14048) — arXiv 2604.14048, 2026. It has the same mechanism as the original S4: test-time LoRA that makes partial-view outputs match the model's own full-view outputs, on VGGT and DA3, in under 2 minutes. It differs in that it is within-sequence or per-dataset recalibration, with no cross-session storage, no recurrent CUT3R state, and no store comparison or policy. It is now the S4 recipe and the TTA-on-B baseline.
- [LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models](https://arxiv.org/pdf/2412.07746) — ICLR 2025. It builds per-scene LoRA adapters (18 MB, 5 minutes) from confidence-calibrated pseudo-labels on DUSt3R/MASt3R. The adapters are evaluated on the same scene and are not used as memory for a later visit or compared with other stores.
- [ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](https://arxiv.org/pdf/2510.11605) — ICCV 2025. It stores a scene in a per-scene map code of a few MB, trained in minutes, on top of a generic transformer, and relocalizes later queries. This is exactly 'scene earned weights for revisits' in the relocalization literature. It is now baseline S6.
- [VGGT-Align: Bridging Local Reconstruction and Global Consistency for Long-Sequence 3D Reconstruction](https://arxiv.org/pdf/2608.15260) — arXiv 2608.15260, 2026. It does LayerNorm-only test-time adaptation for long sequences with self-supervised losses, within one scene. It is the source of the LN-only adapter variant.
- [TTT3R: 3D Reconstruction as Test-Time Training](https://arxiv.org/pdf/2509.26645) — ICLR 2026. It treats the CUT3R state as fast weights with a closed-form learning rate. It is the default writer and the fast-state baseline. It has no cross-session memory.
- [Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction (ABot-Recon)](https://arxiv.org/pdf/2608.27529) — arXiv 2608.27529, 2026. It argues persistent memory is unnecessary outdoors but useful in small, frequently revisited indoor spaces. That bears directly on the regime where the crossover should appear.
- [Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction](https://arxiv.org/pdf/2605.05749) — arXiv 2605.05749, 2026. It detects true loop revisits and refines poses, but only within a single stream, not across sessions.
- [Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction](https://arxiv.org/pdf/2609.04201) — arXiv 2609.04201, 2026. It combines a CUT3R backbone with chunked TTT, place recognition and a pose graph. It is the nearest design to recognizing a mapped area, but within one run.
- [Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training](https://arxiv.org/pdf/2604.07279) — arXiv 2604.07279, 2026. It adds trained fast-weight MLP memory plus token memory, with architecture changes and retraining, and no per-scene cross-session store.
- [Info3R: Information-Adaptive Test-Time Training for 3D Reconstruction](https://arxiv.org/pdf/2609.21938) — arXiv 2609.21938, 2026. A stronger CUT3R state writer that may shrink the overflow regime. It is the no-weights ceiling baseline.
- [TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction](https://arxiv.org/pdf/2601.22615) — arXiv 2601.22615, 2026. Another training-free state writer. It shows how crowded the 'better state' space is.
- [Long-Term Multi-Session 3D Reconstruction](https://arxiv.org/html/2602.20584v1) — arXiv 2602.20584, 2026. It does multi-session reconstruction across years, but offline SfM with cross-session matching, not streaming foundation-model memory.
- [SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning](https://arxiv.org/html/2511.22367v1) — arXiv 2511.22367, 2025. An LLM-side surprise-gated selection with fast/slow LoRA consolidation. It is the analog of the 'earned' signal, not tested on geometric fast state.

**Queries run:** streaming 3D reconstruction revisit scene memory per-scene adapter LoRA CUT3R relocalization; test-time training 3D foundation model persistent scene memory weights across sessions revisit; CUT3R state forgetting long sequence memory consolidation 2026; scene coordinate regression foundation model features per-scene fast training relocalization 2025 2026 ACE MASt3R adapter; multi-session lifelong streaming 3D reconstruction foundation model map reuse revisits CUT3R VGGT; self-distillation feed-forward 3D reconstruction test-time LoRA scene-specific DUSt3R MASt3R fine-tune pseudo-labels; "Free Geometry" refining 3D reconstruction from longer versions of itself; CUT3R state bank store retrieve past states submaps loop closure recurrent state reuse streaming reconstruction; continual test-time adaptation 3D reconstruction model across scenes sequence of scenes catastrophic forgetting adapter library; [extended] streaming 3D reconstruction model returning to previously mapped scene, persist memory across sessions, test-time training weights or state saved and reloaded, CUT3R TTT3R VGGT 2026; "Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction"; deciding when to update weights vs retrieve from memory test-time training trigger signal forgetting score consolidation policy 2026

---

## 8. Do-No-Harm Memory: paired conformal guarantees that self-consolidating agent memory is no worse than no memory, with a rollout-free anchor-likelihood write gate

**Judge:** overall 5.5/10, tier B. Scores out of 5: novelty 3, low compute 2, impact 3, speed of first signal 4.
It is the only plan offering a finite-sample guarantee against the doc's 'memory falls below no-memory' failure (2605.12978), and it works for black-box API actors. That makes it practical, and it is portable to any store. The first signal arrives in two weeks.

**Biggest weakness:** Reviewers may read it as 'apply CRC/ACI to memory'. The write gate competes with Janus, Recuris and AGMR. The plan as written uses Qwen3-8B with vLLM at 160 GPU-hours, which violates the 1.5B/low-compute preference unless moved to API-only. If harm is not predictable, the gate collapses to 'never use memory'.

**One line.** A black-box read gate that decides, per query, whether to inject consolidated text memory. It is calibrated with conformal risk control on paired with-memory and no-memory outcomes, so memory breaks problems the model already solves at a rate of at most alpha, and it is recalibrated online under memory drift. A cheap teacher-forced anchor-NLL write gate stands in for rollout-based consolidation gates such as Janus and Recuris.

**Memory store.** External store (consolidated text memory: a cheatsheet, a playbook or an insight bank), with a raw-episode store as the fallback. The method routes each query to consolidated memory or to no memory, and routes each proposed consolidation either into the consolidated store or into the raw store. It also evicts consolidated entries using their paired win/loss record.

**Task.** Test-time learning with accumulated text memory on verifiable reasoning streams. Calibration and main streams: Game of 24 (1,362 puzzles), MATH level-5 test (about 1.3K), MMLU-Pro Physics (about 1.3K), and a subset of the ARC-AGI-1 training set for the open model. Replication of the original collapse: 200 ARC-AGI-1 eval tasks with memory distilled from ground truth. Transfer-only evaluation: GPQA-Diamond and AIME 2024/25. Stretch goal: ALFWorld unseen (134 games) with a ReasoningBank-style memory, only if budget remains.

**Gap in the big picture.** The big-picture document says 'External memory moves the stability-plasticity dilemma instead of escaping it'. 2605.12978 shows that text memory consolidated continuously by an LLM improves, plateaus, then falls below a no-memory baseline, and that memory distilled from ground truth broke ARC tasks the model had solved without memory. Current remedies come in two kinds. Rollout-based write gates (Janus 2606.31121, Recuris 2608.24876, AGMR 2607.17621) are costly and compare only against the previous memory state. Heuristic read gates (RSCB-MC 2604.27283, the three-signal controller 2609.22043, MemTrim 2610.07311) carry no guarantee. Measurement papers (Memory Trust Gap 2609.01852, NullMemory 2605.17830) quantify harm relative to no memory but do not control it. No system gives a finite-sample or long-run bound on harm relative to the no-memory policy. The write-gate score borrows the document's lesson that forgetting tracks how far an update moves the model (RL's Razor, KL from the base model): it measures the likelihood shift on past competent behaviour. Together the two gates are a concrete, verifiable answer to the open consolidation question of what should move into consolidated memory, and when.

**Hypothesis.** The setting is DC-style memory streams on which ungated cumulative memory falls below the compute-matched no-memory baseline. There, a paired conformal read gate at alpha=0.05 keeps realized paired harm at or below 0.07 and final accuracy at or above the no-memory baseline, while retaining at least 70% of peak memory gain; and the rollout-free anchor-NLL shift flags regression-causing consolidations with AUROC of at least 0.75, at no more than 10% of the token cost of a Janus-style rollout gate.

**Method.**

(1) Paired logging. At each stream step t, with memory M_t from a standard builder (DC-Cumulative, DC-RS, ACE, raw-episode retrieval), run the memory arm once and the no-memory arm k=3 times with short samples. Log both. Memory always evolves from the memory arm's own trajectories. This means every read gate, ours and the baselines, can be evaluated by offline replay at zero extra GPU cost.

(2) Paired harm. h_i = 1[majority no-memory answer is correct AND memory answer is wrong]. Benefit b_i is defined symmetrically. A placebo arm (a second independent no-memory draw) estimates the sampling-noise floor of harm.

(3) Read gate. A logistic harm score s(x) over cheap pre-answer features:
- no-memory self-consistency c0 and the mean token logprob of the majority answer (open model only)
- top retrieval cosine and the number of injected entries
- mean entry age and rewrite count
- the Beta-posterior paired win rate of the injected entries
The gate injects iff s(x) <= lambda; otherwise it deploys the no-memory majority answer.

(4) Offline guarantee. The loss L_i(lambda) = 1[s(x_i) <= lambda] * h_i is monotone in lambda. Conformal risk control picks the largest lambda with (sum_i L_i(lambda) + 1)/(n+1) <= alpha on n of about 300 calibration items, which gives E[harm] <= alpha under exchangeability. A Learn-then-Test variant uses a one-sided binomial test on the net loss (h - b) to certify 'accuracy not below no memory' with probability 1 - delta.

(5) Online guarantee. Memory drift breaks exchangeability. Adaptive conformal inference therefore updates lambda_{t+1} = lambda_t + eta*(alpha - L_t) on a random 10% audit stream where both arms are graded, which bounds long-run average harm without exchangeability. The per-entry paired win/loss counts from audits also drive eviction: an entry whose posterior P(harm rate > benefit rate) exceeds 0.9 is demoted to the raw store.

(6) Secondary write gate. Score each proposed update M_t -> M' by teacher-forcing up to 32 anchor trajectories, sampled with age decay from past memory successes and from 'competence anchors' (tasks the no-memory arm solves). The model is Qwen3-8B via vLLM prompt_logprobs, with the memory as a prefix-cached shared prefix. D = mean answer-span per-token NLL under M' minus the same under M_t, and D0 is computed the same way against no memory. Commit iff the triggering episode's NLL drops and D <= tau_w. Otherwise keep M_t and append the episode raw. Optionally, split M' into atomic diffs and accept the passing diffs greedily. Set tau_w on one dataset against rollout regression labels, then freeze it and transfer it.

**Datasets.**

- Game of 24 (1,362 puzzles; the GameOf24 split in the Dynamic Cheatsheet repo, github.com/suzgunmirac/dynamic-cheatsheet)
- MATH test, level-5 problems only (EleutherAI/hendrycks_math, about 1.3K)
- MMLU-Pro Physics and Engineering (TIGER-Lab/MMLU-Pro; both supported in the DC repo)
- GPQA-Diamond (198; transfer-only evaluation)
- AIME 2024 + AIME 2025 (60; transfer-only evaluation)
- ARC-AGI-1 (fchollet/ARC-AGI): 200-task eval subset for the API actor with memory distilled from ground truth; a training-split subset for Qwen3-8B
- Stretch: ALFWorld unseen split (134 games)

**Models / checkpoints.**

- Qwen/Qwen3-8B (main open actor in non-thinking mode, plus a thinking-mode ablation on MATH-L5; also the anchor-NLL scorer via vLLM prompt_logprobs)
- Qwen/Qwen3-4B (capability ablation, following the capability-dependent harm finding of Memory Trust Gap 2609.01852)
- Qwen/Qwen3-Embedding-0.6B (retrieval for DC-RS, raw-episode retrieval and the similarity features)
- claude-haiku-5-5 via the Anthropic API (check current pricing; black-box actor; read gate only, because there are no logprobs; the write gate uses Qwen3-8B as a surrogate scorer)
- gpt-5-mini via the OpenAI API (optional second black-box actor for the ARC replication)

**Baselines.**

- Memory builders, run ungated: Dynamic Cheatsheet-Cumulative and DC-RS (official code), ACE generator/reflector/curator (official code at github.com/ace-agent/ace, with a Qwen3-8B backend), raw-episode top-k retrieval (the 'Episodic-only' regime of 2605.12978)
- No memory, compute-matched (self-consistency over 4 samples), and always-inject memory with self-consistency over 4 samples
- Read-gate baselines: similarity threshold; Self-RAG-style self-assessment prompt; RSCB-MC risk-sensitive LinUCB with asymmetric penalty (reimplemented, 2604.27283); three-signal confidence/consistency controller (reimplemented, 2609.22043); MemTrim (2610.07311) if code is released
- Ablations of the guarantee: plug-in threshold on the same features without the conformal correction; C-RAG-style conformal bound on absolute memory-arm risk instead of paired harm; offline CRC without ACI under drift
- Write-gate baselines: always consolidate; never consolidate (raw only); Janus-style rollout gate with a boundary-task support set and a momentum trigger (reimplemented, 2606.31121); Recuris-style rollout gate that checks fixes plus no regression on a dev set (2608.24876); InfoMem-style gold-answer likelihood gain over empty memory as the gate score (2606.03329); a random gate at a matched acceptance rate

**Metrics.**

- Accuracy vs stream position (the improve-plateau-drop curve) and area under it, both relative to the compute-matched no-memory baseline
- Paired harm rate P(no-memory correct, deployed wrong), paired benefit rate, and net gain; harm minus the placebo noise floor
- Realized vs nominal alpha over 200 random calibration/test splits (offline CRC), and the running average harm (online ACI)
- Share of memory gain retained = (acc_gated - acc_nomem) / (acc_peak_mem - acc_nomem), and the injection (non-abstention) rate
- Write gate: AUROC and AUPRC of D against rollout regression labels; regression rate on previously solved tasks; committed-update rate
- Cost: generated vs prefilled tokens and GPU-seconds per write decision, relative to the Janus/Recuris rollout gates; audit token overhead of the read gate
- Feature attribution: AUROC of no-memory self-consistency alone for harm (hypothesis that memory overrides existing competence)

**First experiment (go/no-go).**

Weeks 1-2, one H100, about 25-30 GPU-hours plus under $30 of API.

Runs:
- Run DC-Cumulative and raw-episode retrieval with Qwen3-8B (non-thinking) on the Game of 24 and MATH-L5 streams, 2 seeds each, under vLLM. Run several streams concurrently to fill the batch.
- Log the memory arm (1 sample), the no-memory arm (3 samples) and a placebo no-memory draw for every item.
- Compute anchor-NLL D (16-32 anchors, prefix-cached) for every proposed update.
- In parallel, run claude-haiku-5-5 on the 200-task ARC-AGI-1 eval subset with memory distilled from ground truth, to replicate the ARC breakage reported in 2605.12978.

Analysis, all on CPU:
- (i) Accuracy-vs-position curves against compute-matched no memory.
- (ii) Late-stream paired harm minus the placebo floor.
- (iii) 5-fold CV AUROC of the logistic harm score on the cheap features.
- (iv) Offline CRC at alpha=0.05 over 200 random splits: realized harm and retained gain.
- (v) For 200 sampled updates, rollout labels from re-running 32 previously solved probe tasks under M' vs M_t (2 samples each), and the AUROC of D against these labels.

GO if all of the following hold:
- Paired harm above the noise floor is at least 5 points late in the stream on at least one stream (Qwen3-8B or Haiku).
- Harm AUROC is at least 0.70.
- CRC meets alpha while retaining at least 50% of the memory gain.

The write gate is kept only if the AUROC of D is at least 0.65.

**Kill criterion.**

Drop the paper, or downgrade it to a short measurement note, if after week 2 either of these holds:
- (a) Paired harm minus the placebo noise floor is under 2 points on every stream and actor, including Haiku on ARC, so there is nothing to control.
- (b) Harm is not predictable from pre-answer features (AUROC under 0.62), so alpha=0.05 can only be met by abstaining on more than 80% of queries, retaining under 30% of the memory gain. The method would then collapse to 'never use memory'.

Independently, cut the write gate and publish the read-gate guarantee alone if the AUROC of D against rollout regression labels is under 0.65, or if the cost-matched comparison with the Janus-style gate shows at least 10 points lower regression-rejection accuracy.

**Compute.** 160 GPU-hours over 14 weeks. One NVIDIA H100 80GB (an A100 80GB works at about 1.4x the wall time) running vLLM for Qwen3-8B, Qwen3-4B and Qwen3-Embedding-0.6B, with about 8-16 streams batched concurrently. All gates, calibration (CRC, LTT, ACI) and analysis run on CPU. API spend is about $200-400 in total (claude-haiku-5-5 as the main black-box actor, gpt-5-mini optional).

Estimated GPU-hours:
- About 12: week-1 reproduction.
- About 30: main paired streams (4 datasets x 3 builders x 3 seeds, 8B).
- About 20: anchor scoring.
- About 15: rollout regression labels.
- About 10: Janus- and Recuris-style rollout-gate runs.
- About 35: runs with the write gate in the loop.
- About 16: 4B and thinking-mode ablations.
- About 20: slack.

**Risks.**

(1) No phenomenon with open models. Qwen3-8B may not reproduce the drop that 2605.12978 reported with GPT-5-class models. Mitigation: Haiku/gpt-5-mini on ARC, the Qwen3-4B capability ablation, and ARC memory distilled from ground truth. If harm stays at the noise floor everywhere, the kill criterion applies.

(2) Label noise. Recuris found that re-running an identical package broke 25.9% of tasks, so sampling noise can masquerade as harm. Mitigation: majority-of-3 no-memory labels, placebo-arm noise floor, and harm reported net of the floor.

(3) Compute fairness. The gate spends 3 no-memory samples, so all comparisons use compute-matched sc@4 baselines. A 1-sample variant uses logprob confidence instead.

(4) Exchangeability. Memory drift breaks it, so offline CRC holds only within stationary windows. ACI gives only long-run guarantees. Mitigation: report both, plus a stream-position-weighted conformal variant.

(5) Format, not behaviour. Answer-span NLL may track format rather than behaviour. Mitigation: length-normalized answer-span scoring and a format-control span.

(6) Black-box actors. The API actors return no logprobs, so the write gate relies on a Qwen3-8B surrogate scorer, whose transfer must be tested.

(7) Audit labels. Online audits need ground truth on 10% of queries. This is trivial for Game of 24 and MATH, but a real limitation for open-ended agents.

(8) Novelty and timing. This area moves weekly. MemTrim (Oct 5 2026) and the three-signal controller are concurrent. Lead with the paired guarantee and submit fast.

(9) Framing. Reviewers may read the read gate as RAG abstention. The paired 'relative to no memory' loss and the C-RAG-style absolute-risk ablation must be central.

(10) Unverified claims. Several 2026 arXiv claims (including the Janus internals) could not be opened during the novelty check and must be verified before submission.

**Target venue.** ICML 2027 (deadline about late January 2027, which fits a 14-week plan from 2026-10-08); fallback NeurIPS 2027 or TMLR

**First-round prior-work verdict: partially_novel.** Rejecting consolidations that regress past solved tasks, and falling back to raw episodes, is already done with rollouts by Janus (2606.31121), Recuris (2608.24876) and AGMR (2607.17621), and recommended by 2605.12978. Heuristic read gates with abstention also exist (RSCB-MC, the three-signal controller, MemTrim). The defensible delta is the lead contribution: a paired, finite-sample (CRC/LTT) and online (ACI) bound on harm relative to the no-memory policy, which none of these provide and which works for black-box API actors. A secondary, cost-matched claim is that a rollout-free anchor-NLL shift, including competence anchors that Janus-style support sets omit, matches rollout gates at a fraction of the cost. Before claiming the latter, check the Janus PDF for any likelihood proxy. Arxiv pages could not be opened during the novelty check, so overlap claims rest on search snippets.

**Closest work found in the first round:**

- [The Past Is Prologue: A Plug-in Controller for Selective Updates in Sequentially Evolving LLM Memory (Janus)](https://arxiv.org/abs/2606.31121) — arXiv 2606.31121, Jun 2026. A method-agnostic write gate that wraps existing memory updaters (it was tested with two updaters on six datasets) and accepts a candidate memory only if it beats the old one on a hybrid batch. The batch combines a stored support set, made of coverage tasks plus 'boundary tasks where memory choices previously changed correctness', with fresh recent tasks. A momentum trigger decides when to run the check. This is very close to the idea's anchor-set write gate. The difference is that Janus scores by executing the tasks (rollouts), not by teacher-forced likelihood, and it gives no guarantee relative to no memory. The idea already names it as the 'Janus-style rollout gate' baseline.
- [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) — arXiv 2605.12978, May 2026. This is the motivating paper. It already proposes a remedy: gate consolidation explicitly and keep raw episodes as first-class evidence (the Auto and Episodic-only regimes recover most of the gain). The idea's fallback 'keep M_old and store the episode raw' is close to this recommendation. The paper does not propose a likelihood score or a conformal harm bound.
- [Recursive Experiential-Working Memory Evolution for Long-Horizon Agent Harnesses (Recuris)](https://arxiv.org/pdf/2608.24876) — arXiv 2608.24876, Aug 2026. A fixed validation gate admits a memory patch only if it repairs the source tasks and does not regress a held-out dev set. This is the same 'helps the triggering episode and doesn't hurt anchors' rule as the idea's write gate, but it is rollout-based. The paper also reports that its gate mostly filters noise (re-running an identical package broke 25.9% of tasks). That supports the idea's concern about sampling noise.
- [Mechanistic Attention Guidance for Agent Memory Refinement (AGMR)](https://arxiv.org/abs/2607.17621) — arXiv 2607.17621, Jul 2026. Uses an internal model signal (attention from retrieval heads over memory segments) rather than text alone to guide memory edits. It edits at the segment level and re-executes every candidate update before committing, reverting if the retries fail. It overlaps with the idea's use of model internals and its atomic-edit testing, but it uses attention and rollouts, not NLL shift on anchors.
- [InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain](https://arxiv.org/abs/2606.03329) — arXiv 2606.03329, Jun 2026. Scores a memory by the teacher-forced per-token log-likelihood of gold answers with that memory minus the same quantity with an empty memory. This is essentially the idea's rollout-free memory-vs-no-memory likelihood score. InfoMem uses it as an RL training reward on successful trajectories, not as a write gate against regressions.
- [Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents (RSCB-MC)](https://arxiv.org/abs/2604.27283) — arXiv 2604.27283, Apr 2026. A read gate that chooses to inject, summarize or abstain, and penalizes false-positive memory injection more heavily than missed reuse. That is the same read-time decision as the idea. It gives no finite-sample or conformal guarantee, and its evaluation is a smoke-scale proxy on coding tasks. The idea already cites it.
- [An Interpretable Memory Decision Controller for LLM Agents Based on Three-Signal Complementarity: Decoupling Confidence and Consistency](https://arxiv.org/html/2609.22043) — arXiv 2609.22043, Sep 2026. A read-time memory trust controller that combines confidence and consistency signals and has explicit abstention. It overlaps with the idea's feature-based read gate (self-consistency plus retrieval signals). From the snippets, it has no conformal guarantee and does not frame harm relative to no memory. The idea does not cite it.
- [Understanding and Mitigating Inference-Time Overreliance Using Agentic Memory (MemTrim)](https://arxiv.org/abs/2610.07311) — arXiv 2610.07311, Oct 5 2026. A concurrent paper on 'memory overreliance': accurate, correctly retrieved memories still distort reasoning under partial overlap. It proposes a write-time indexing plus read-time control method that is plug-and-play. This is the same thesis as hypothesis (3), that memory overrides competence the model already has. It is heuristic, with no guarantee. The idea does not cite it.
- [Mitigating LLM Hallucinations via Conformal Abstention](https://arxiv.org/abs/2405.01563) — arXiv 2405.01563, 2024. Calibrates an abstention threshold with conformal methods to bound the error rate. It is a standard template for the idea's CRC read gate, but it acts on answers and bounds absolute risk, not paired harm relative to no memory.
- [Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents](https://arxiv.org/html/2605.17830v1) — arXiv 2605.17830, May 2026. Uses a NullMemory counterfactual baseline to define memory-induced harm, paired against the same agent without memory. It is a safety measurement only, with no control, but it pre-empts part of the 'harm relative to no memory' framing.
- [The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents](https://arxiv.org/pdf/2609.01852) — arXiv 2609.01852, Sep 2026. Measures paired net harm of memory relative to no memory on Qwen3 0.6B to 8B and finds the harm depends on capability. It measures but does not control, as the idea already says.
- [MonoScale: Scaling Multi-Agent System with Monotonic Improvement](https://arxiv.org/pdf/2601.23219) — arXiv 2601.23219, Jan 2026. Gives a monotonic non-decreasing performance guarantee for a memory update protocol, but under bandit and non-interference assumptions and relative to the prior memory state, not to no memory. It is a partial precedent for 'guaranteed no worse' memory updates.

---

## 9. Rent, Buy, or Let It Expire: Competitive Consolidation of Changing Facts from a Retrieval Store into a Small LM

**Judge:** overall 5.5/10, tier B. Scores out of 5: novelty 3, low compute 3, impact 4, speed of first signal 3.
It frames the doc's open problem (what moves into the weights, and when) as an online rent-or-buy problem with expiring purchases and store externalities, which is an elegant framing. The theory and simulation run on CPU, and the plan fits arc 2 next to P9 and P10.

**Biggest weakness:** Building a real Wikidata revision stream is heavy engineering. Small-model writes (MEMIT/LoRA on 0.5B) may give poor paraphrase generalization, so the honest answer may be 'never buy'. ALLOT and Dual-Layer already learn which facts to write, and ALLOT's method section is still unread. The headline result arrives late.

**One line.** Moving a fact that keeps changing from a RAG store into a 0.5-1.5B LM's weights is framed as an online rent-or-buy problem with two twists: a purchase expires when the fact changes, and buying a fact shrinks the store, which lowers the rent of every other fact. Each purchase is priced by a forecast of its KL drift made before the write, and the policy is scored against the clairvoyant schedule on a real Wikidata change stream with pageview-driven demand.

**Memory store.** Cross-store. Facts move from the external retrieval store into the weights through MEMIT, AlphaEdit or LoRA micro-updates. A fact is evicted from the store once consolidated. When its value changes it is demoted back to retrieval-first, and the next buy decision starts a new epoch.

**Task.** Keep a 0.5-1.5B LM current over 24 monthly steps of real fact changes and pageview-weighted queries.
- Facts: about 8k (subject, relation) pairs over about 12 Wikidata relations. Volatile relations: P54 sports team, P286 head coach, P6/P35 head of government/state, P488 chairperson, P169 CEO, P102 party. Stable controls: P36 capital, P19 birthplace, P27 citizenship, P50 author, P175 performer.
- Value sequences are rebuilt month by month from Wikidata revision history on CPU. Windows: 2024-01 to 2025-12 for Qwen2.5; 2021-01 to 2022-12 for Pythia and GPT-2 XL, which suits their 2020 data.
- Query demand comes from Wikimedia monthly pageviews of the subject.
- Pre-write cost forecasting is studied on CounterFact, zsRE and WikiFactDiff replacements.
- Retention is measured on NQ-open and TriviaQA subsets and WikiText-103 perplexity.
- A synthetic world on CPU, with per-relation hazards and Zipf demand, covers the theory checks.

**Gap in the big picture.** This tackles the document's central open problem: what should move from an external store into the weights, and when. Three lessons from the document become cost terms:
- Forgetting tracks how far an update moves the model (RL's Razor KL; sparse memory finetuning, 2510.15103). That KL becomes the buy price, but it is known only after writing, so the project forecasts it beforehand.
- When continual learning moves to memory, old and new items compete at retrieval (2604.27003). This becomes a rent per query that grows with store size N, so each consolidation and eviction lowers every other fact's rent.
- Harrington et al. (2607.07847) find that the kind of change decides which store should adapt. Here the kind of change becomes a measurable per-relation change hazard that makes purchases expire.

Recent work (ALLOT, Dual-Layer, EVAF) learns statically which facts get weight writes. None models query-rate rent, fact expiry, store-size externalities or the gap to a clairvoyant schedule over time.

**Hypothesis.** On a 24-month Wikidata change stream with pageview-weighted queries, a learning-augmented ski-rental policy will stay within 2x of the clairvoyant optimum's total cost when run on Qwen2.5-0.5B. The policy buys a fact into the weights once the rent accumulated since its last change exceeds a hazard-discounted, pre-write forecast of its KL-drift price. Against a budget-matched ALLOT-style static router, it will cut total cost by at least 15% and stale answers by at least 30%.

**Method.**

1) Cost model, all terms measured on the actual LM:
- A fact held externally pays rent per query: rho(N) = c_tok * context tokens + c_err * e(N). The retrieval error e(N) is fitted empirically for BM25 and bge-small at N in {1k, 5k, 20k, 50k} with same-relation distractors.
- Buying costs B(f) = c_comp * write compute + lambda * D(f). D is the realized drift: token-level KL(base || edited) on a fixed anchor set (1k WikiText-103 sequences plus 500 NQ prompts) plus accuracy loss on neighbouring facts.
- A purchase expires when the fact changes. The hazard h_r for relation r is a per-relation Weibull fitted to Wikidata edit history. A stale weight copy is charged a conflict penalty, measured as the context-faithfulness failure rate when the new value is retrieved against the old parametric prior.

2) Pre-write price forecast B_hat(f), computed without committing any update:
- Base-model log-prob margin of o_old over o_new.
- An off-policy gap: log p(o_new|x) minus the mean log-prob of the model's own 8 samples. This is a training-free proxy for RL's Razor.
- Parametric conflict: whether the greedy answer is o_old.
- infini-gram Pile counts of (s, o_old) co-occurrence.
- Subject pageviews.
- Store crowding: the number of store items with bge cosine similarity above tau.
- A dry-run signal: the MEMIT residual ||v* - W k*||, or the gradient norm at the edit layer.
These feed a small ridge or GBM model trained to predict realized D.

3) Policy: buy f in the monthly batch when the rent accumulated in the current epoch exceeds beta * B_hat(f) * phi(h_r). The discount phi falls with hazard, Bahncard-style.
- A learning-augmented variant (Purohit et al. 2018) trusts the predicted query rate q_hat, hazard h_hat and B_hat with confidence lambda.
- Theory, all checkable on CPU: (a) per-epoch 2-competitiveness and consistency/robustness bounds for the decoupled case; (b) under Poisson demand and exponential expiry, the Bayes-optimal rule collapses to a static index q*rho(N)/h >= B; (c) a mean-field treatment of the eviction externality predicts that the optimal threshold falls as N grows.

4) The clairvoyant optimum is computed two ways: per-fact dynamic programming over realized epochs, and an exact ILP with the N-coupling (HiGHS) on 1-2k-fact subsets. The empirical competitive ratio is reported against both.

5) Pythia natural experiment: edit the same facts at intermediate checkpoints. A difference-in-differences of write cost compares facts with high and zero Pile counts.

**Datasets.**

- Real monthly change stream built on CPU from Wikidata revision history (wbgetentities/revisions API or dumps): about 8k (s, r) pairs over about 12 relations, 24 steps
- Wikimedia Pageviews API, monthly per-article views, as query demand
- WikiBigEdit (github.com/ExplainableML/WikiBigEdit): question templates, rephrasings and locality probes for Wikidata updates
- TemporalWiki TWiki-Diffsets/Probes (HF seonghyeonye/TemporalWiki, monthly Aug-Dec 2021): secondary short stream with changed and unchanged probes
- WikiFactDiff (HF OrangeInnov/WikiFactDiff, Jan 2021 to Feb 2023 replacement updates with neighbour triples): forecasting facts for Pythia and GPT-2 XL
- CounterFact (ROME/MEMIT release): neighbourhood and paraphrase prompts for drift and damage
- zsRE (MEND/EasyEdit split)
- infini-gram Pile-train index (API) for pretraining co-occurrence counts
- NQ-open and TriviaQA (1k-question subsets) for retention; WikiText-103 for perplexity and KL anchors
- Synthetic world (CPU): per-relation Weibull hazards, Zipf demand, N-dependent retrieval error

**Models / checkpoints.**

- Qwen/Qwen2.5-0.5B-Instruct (main reader and writer)
- Qwen/Qwen2.5-1.5B-Instruct (scale check)
- EleutherAI/pythia-410m and EleutherAI/pythia-1.4b (non-deduped, to match the infini-gram Pile-train index) with revisions step13000/step33000/step63000/step93000/step143000
- openai-community/gpt2-xl (fallback writer target; EasyEdit/AlphaEdit ship MEMIT and AlphaEdit hparams for it)
- BAAI/bge-small-en-v1.5 (dense retriever); BM25 via pyserini or rank_bm25
- Editors: MEMIT and AlphaEdit through EasyEdit (zjunlp/EasyEdit ships AlphaEdit configs for gpt2-xl and qwen2.5-7b; layers and covariance stats must be re-derived for 0.5B/1.5B); LoRA r=8-16 micro-updates through PEFT

**Baselines.**

- RAG-only (never consolidate)
- Consolidate-all (write every change as it arrives)
- LFU-k and LRU-k write-back (top-k queried facts each month, k matched to the policy's write budget)
- ALLOT-style budgeted static router, re-implemented: frozen bge embeddings, retrieval-neighbourhood statistics and relation one-hot into logistic regression, budget-matched (arXiv 2609.32344)
- Dual-Layer-style periodic SFT write-back with an 'already internal' suppression gate (arXiv 2608.22215)
- EVAF-style surprise-gated selective LoRA consolidation (arXiv 2606.29916)
- Random consolidation at matched budget
- Continual fine-tuning on monthly diffs (WikiBigEdit protocol)
- Ablations of the proposed policy: no hazard term, no externality term, Knowledge-Spectrum features only (popularity + known/unknown) as the forecast, and oracle predictions (upper bound)
- Clairvoyant optimum (per-fact DP and coupled ILP on subsets)

**Metrics.**

- Total cost (rent + buy + stale/conflict penalty), normalized
- Empirical competitive ratio vs per-fact DP and coupled-ILP optima
- Current-fact accuracy (alias-matched EM on rephrased questions)
- Stale-answer rate
- KL drift on the anchor set; WikiText-103 perplexity change
- NQ-open and TriviaQA EM retention
- Context tokens per query; store size N over time
- Writes per month and GPU-seconds spent writing
- Forecast quality: cross-validated Spearman and AUC (top-quartile cost) of B_hat against realized KL and neighbour damage, plus incremental Spearman over Knowledge-Spectrum features
- Retrieval recall@k as a function of N (fitted e(N))
- Pythia: difference-in-differences in write cost across checkpoints, high- vs zero-Pile-count facts

**First experiment (go/no-go).**

Two tracks run in weeks 1-2 on one GPU plus CPU.

(A) Price forecasting on Qwen2.5-0.5B-Instruct:
- Take 2,000 CounterFact candidates and keep those the model answers correctly in its top-5, aiming for about 500 facts. If too few survive, add Qwen2.5-1.5B or WikiFactDiff facts with GPT-2 XL.
- Compute the pre-write signals: margin, off-policy gap with 8 samples, conflict, infini-gram counts, pageviews, crowding, and the MEMIT dry-run residual.
- Apply each edit alone with MEMIT and with LoRA r=8 (20 steps). Covariance stats take about 1 GPU-hour.
- Measure realized KL on 1k anchor sequences and neighbourhood accuracy drop.
- Cost: about 15 GPU-hours.

(B) CPU only: the synthetic world with 10k facts, 3 hazard regimes and Zipf demand. Compare the threshold policy, the ALLOT-style static budgeted ranker, LFU and consolidate-all against the exact optimum, and check whether the optimal threshold falls with N.

Go if all of the following hold:
- The cross-validated combined forecaster reaches Spearman >= 0.4 against realized KL for at least one writer.
- It adds >= 0.1 Spearman over popularity + known/unknown features alone.
- Realized prices vary enough to matter: the 75th/25th percentile ratio of realized B is >= 3.
- In simulation, the policy stays within 2x of the optimum while the static ranker is >= 1.3x worse than the policy in the high-volatility regime.

**Kill criterion.**

Each condition below triggers a specific fallback or stop.
- Forecast fails: no pre-write signal set reaches cross-validated Spearman >= 0.3 with realized KL on either writer, or realized write costs are nearly constant across facts (75th/25th percentile ratio < 2). Drop the forecasting contribution and fall back to a CPU theory-plus-simulation paper, with a constant price measured on a small set.
- Buying never pays off: on the first 6 months of the real stream, buying never beats rent because small-model writes reach under 50% paraphrase accuracy, or because stale-copy conflicts dominate. Report 'never buy' as the finding for 0.5-1.5B models.
- No gain over the static router: the threshold policy fails to beat the budget-matched ALLOT-style router by at least 10% total cost or 15% stale-answer rate. Drop the empirical policy claim.
- Scooped: the ALLOT full text turns out to include query-rate rent, expiry or online guarantees. Stop.

**Compute.** 160 GPU-hours over 15 weeks. One A100-80GB or L40S-48GB; a 24 GB card is enough for all models of 1.5B parameters or fewer. CPU covers the Wikidata history crawl, pageviews, the synthetic world, the DP/ILP optima (HiGHS) and the infini-gram API calls.

Budget breakdown, about 160 GPU-hours in total:
- Covariance stats: about 6
- Forecasting study (2 models x MEMIT/AlphaEdit x 1k facts, plus LoRA x 500 facts): about 45
- Pythia checkpoint study: about 25
- Stream runs (9 policies x MEMIT on 0.5B, 4 policies x LoRA, 4 policies on 1.5B): about 60
- Retrieval curves, signals, debugging and reruns: about 25

**Risks.**

- ALLOT's full method is unread and may already cover part of the temporal or demand delta.
- Sequential-edit collapse and non-additive interference between writes (2607.11020) make B non-stationary. The mitigation is a cumulative-edit degradation term in B_hat, fitted on the first months, but it weakens the clean per-epoch bound.
- Knowledge injection into 0.5-1.5B models is weak (Ovadia et al. 2024), and paraphrase generalization of MEMIT/LoRA may be poor, so the policy could degenerate to 'never buy volatile relations'. That result is still reportable, but it is less exciting.
- MEMIT/AlphaEdit layer choice and covariance stats for Qwen2.5-0.5B/1.5B must be re-derived; GPT-2 XL with official configs is the fallback.
- Building the monthly stream from Wikidata history is engineering-heavy, and aliases and qualifiers make answer matching noisy.
- Pageviews are only a proxy for query demand.
- Qwen2.5's knowledge cutoff is not documented, so some 'new' values may already be known. Probe this before the stream starts.
- With N-coupling the true optimum is an ILP, so the competitive ratio on the full stream is reported against a mean-field or subset optimum.
- Reviewers may see the competitive theory as textbook. The contribution has to rest on the expiry and externality analysis plus the measured gaps.

**Target venue.** ICML 2027 (deadline about late January 2027, which fits the 15-week plan); fallback CoLM 2027 or TMLR

**First-round prior-work verdict: partially_novel.** ALLOT (2609.32344) already learns from pre-write features which facts get a LoRA write on top of external memory, under a static budget and compared with a budget-matched oracle. Dual-Layer (2608.22215) already uses a write router with periodic consolidation into weights, and EVAF already does surprise-gated LoRA consolidation. 'Learned selection of facts to write' is therefore taken, and this plan repositions around what remains:
- the online rent-or-buy formulation with expiring purchases (per-relation hazards from Wikidata history) and store-size externalities from eviction;
- competitive bounds and the empirical ratio against a clairvoyant optimum over time on a real change stream;
- forecasting realized KL drift and neighbour damage, not edit success as in Knowledge Spectrum;
- the Pythia checkpoint difference-in-differences.

Caveats:
- The web-search budget ran out and arxiv.org was unreachable, so ALLOT's method section is still unread. Confirm before week 3 that it has no demand, expiry or online component.
- The released WikiFactDiff has only two snapshots (Jan 2021 and Feb 2023), so the monthly stream has to be built from Wikidata revision history.

**Closest work found in the first round:**

- [ALLOT: Budgeted Hybrid-Memory Routing for Knowledge Updates in LLMs (arXiv 2609.32344)](https://arxiv.org/abs/2609.32344) — arXiv 2026 (Sept). The most threatening hit, and very recent. ALLOT decides which factual updates get a parametric write: every fact is written to external memory, and only facts the router ranks highly also get a LoRA adapter, under a fixed write budget. The router uses frozen text embeddings, retrieval-neighbourhood statistics (similar to the idea's 'store crowding' signal) and relation metadata. It is evaluated on CounterFact with Qwen3-4B and reports the share of a budget-matched oracle's gain it recovers. So 'learn which facts deserve weight writes, from pre-write features, and compare to an oracle' already exists. From the abstract and intro I saw, it is a static, budgeted ranking. I found nothing about query-rate rent, fact expiry or volatility, eviction after consolidation, KL-drift price forecasting, or online competitive analysis. The method section was not readable, so that is uncertain.
- [Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation (arXiv 2608.22215)](https://arxiv.org/abs/2608.22215) — arXiv 2026 (Aug). Manages the external-to-weights lifecycle. A cascade router (1.7B/8B) labels incoming facts as non-write, write-new or write-update. Periodic SFT write-back then internalizes selected external memories, and the router learns to suppress writes for facts that are already internal. The paper frames the problem as external-store bloat (retrieval degradation and cost) versus forgetting from weight writes, which is the same tension as the idea's rent versus buy. In the snippets I saw, write-back has no explicit cost model, query-rate accounting, volatility or expiry term, or competitive guarantee. I could not confirm whether consolidated entries are evicted.
- [EVAF: A Test-Retest Protocol for Selective Parametric Consolidation (arXiv 2606.29916) and companion 'Memory Depth, Not Memory Access' (arXiv 2606.26806)](https://arxiv.org/pdf/2606.29916) — arXiv 2026 (June). Selective LoRA consolidation of experiences into weights, gated by surprise and valence, with retrieval kept as a separate path. It writes only 2-3 times per 200 events, on small models (GPT-2, TinyLlama, Mistral-7B). The companion paper flags stale-memory invalidation (fact change and update) as unsolved. This covers the 'surprise-gated consolidation' idea and the small-model setting, but it uses no cost or rent model and no volatility-based expiry. Its target is persona and goal persistence, not keeping facts current.
- [Diagnosing Model Editing via Knowledge Spectrum (arXiv 2509.17482)](https://arxiv.org/html/2509.17482v1) — arXiv 2025. Pre-edit signals (subject popularity from Wikipedia pageviews, whether the model already knows the fact, question type) predict editing success and stability. Overwriting known facts is harder and less safe than inserting unknown ones. A Knowledge-Diagnostic Framework sets editing intensity from the predicted difficulty. This partly covers hypothesis (1): pre-write signals forecast write difficulty, and the base-model margin and parametric-conflict signals are close relatives of 'known vs unknown'. It predicts edit success rather than realized KL drift or neighbour damage, and it does not feed the forecast into a consolidation policy.
- [LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations (arXiv 2509.03405)](https://arxiv.org/pdf/2509.03405) — arXiv 2025. Links pretraining-data exposure and training dynamics (prediction brittleness across checkpoints) to how editable a fact is. This overlaps the idea's 'pretraining exposure raises write cost' experiment with Pythia checkpoints. LMEnt studies in-context editing on its own models rather than parameter edits at each checkpoint, so the causal checkpoint experiment the idea proposes remains open.
- [Linear Elastic Caching via Ski Rental (Google, CIDR 2025)](https://research.google/pubs/linear-elastic-caching-via-ski-rental/) — CIDR 2025. Applies a learned ski-rental policy to cache residency, as rent versus evict with a miss penalty, in Spanner. It shows that learning-augmented ski rental for memory tiering is established algorithmic practice. It has no connection to LLM weights or knowledge.
- [WikiBigEdit (arXiv 2503.05683) / WikiFactDiff (arXiv 2403.14364)](https://arxiv.org/html/2503.05683v2) — ICML 2025 / LREC-COLING 2024. Real Wikidata fact-change streams for lifelong updating. WikiBigEdit compares editing, retrieval augmentation and continual fine-tuning on these streams but proposes no routing or consolidation policy. These are benchmarks the idea would use, not competing methods.
- [RL's Razor: Why Online Reinforcement Learning Forgets Less (arXiv 2509.04259)](https://arxiv.org/pdf/2509.04259) — ICLR 2026. Forgetting tracks the KL between the fine-tuned and base policy on the new task, measured after training. This is the post-hoc quantity the idea wants to forecast before writing. I found no paper that turns the off-policy gap into a training-free, pre-write forecast.

---

## 10. Overwrite or Wait? Quickest-Change-Detection Belief Slots and an Entrenchment Law for LLM Agent Memory

**Judge:** overall 5.5/10, tier B. Scores out of 5: novelty 3, low compute 4, impact 3, speed of first signal 5.
The CPU-only first week gives a fast signal. The entrenchment law, under which Memini/Benna-Fusi-style coupled fast/slow memories entrench old facts linearly in repetition count, is the piece that ties it to the doc's multi-timescale bet. It is a good theory companion to P13.

**Biggest weakness:** A Bayesian change-point filter on slots may look obvious, and a tuned k-confirm heuristic may match it. The plan depends on LLM stance and scope extraction quality. BBT, MemOps, BeliefMem and MemTxn crowd the method side, and the 7B serving cost is avoidable but listed.

**One line.** Treat every memory overwrite as a Bayesian quickest-change decision. Each slot runs a change-point filter in which reliability cues set the observation noise and the attribute class sets the hazard. Temporary scopes open sub-states that expire, and superseded values are kept as versions. A CPU-checkable law shows that multi-timescale consolidation entrenches old facts roughly linearly in how often they were repeated.

**Memory store.** External store (structured slot memory). A secondary diagnostic covers entrenchment across context, RAG, LLM-consolidated text memory, Memini-style multi-timescale graph memory and LoRA weights.

**Task.** Long-term conversational memory of evolving personal and world facts, i.e. knowledge update and conflict resolution. A cue's value changes over time, and the observations of that change may be clean, hedged, hearsay, retracted or temporary ('in Lyon this week'). Queries ask for the current, previous or first value, or how many times the value changed.

**Gap in the big picture.** The doc's lesson that external memory moves the stability-plasticity dilemma instead of escaping it (2604.27003) applies at one exact point: the overwrite decision. Latest-wins versioning gives maximal plasticity, so any hedged, hearsay or retracted remark flips the stored fact. Repetition-based consolidation, the doc's multi-timescale bet (Memini's coupled fast/slow edges), gives entrenchment: a fact repeated R times resists a real change. LLM-driven consolidation makes this decision implicitly, with no accountability, and degrades over time (2605.12978). Harrington et al. (2607.07847) find that online methods are fragile under noise and that the kind of change determines what adaptation is needed. Several systems touch parts of this. BeliefMem uses Noisy-OR beliefs, MemTxn uses evidence-gated commits, the freshness recipe in 2606.01435 uses latest-wins with history, and BBT uses a Dirichlet-EMA with gates. None of them treats the overwrite as calibrated inference that separates noise from real change and permanent change from temporary. None explains analytically why repetition-weighted multi-timescale memories entrench. The slot posterior also gives a principled answer to the doc's open consolidation question, 'what is stable enough to move, and when', which we report as a by-product signal.

**Hypothesis.** On noisy, tentative and retracted updates (MemOps recency_trap, the implicit-staleness benchmark, and our R x noise grid), a per-slot Bayesian change-point gate at least halves the false-update rate of same-extractor latest-wins slots and of Mem0 at equal or lower update latency. It stays within 2 points of latest-wins on clean MemoryAgentBench FactConsolidation-SH and LongMemEval-S knowledge-update. Its update latency stays bounded as prior reinforcement count R grows, whereas Memini-style fast/slow consolidation's latency grows roughly linearly in R.

**Method.**

(1) Write: Qwen2.5-7B-Instruct, the only LLM step, extracts tuples from each session: (entity, attribute, value, stance in {asserted, hedged, hearsay, retracted}, scope in {permanent, temporary + expiry hint}, event time). Entities and attributes are canonicalized with bge-small embeddings into about 12 attribute classes (location, employer, relationship, preference, possession, schedule, count, ...). (2) Keep: each (entity, attribute) slot runs Bayesian online change-point detection (Adams-MacKay) over run length and current value. Observations are categorical: the observed value is the true segment value with probability 1 - eps_s, where the noise rate eps_s depends on stance and is fit on a held-out calibration split. The per-session hazard h_c has a Beta prior per attribute class and is updated empirically from the slot's own change history. (3) Write gain: for a held value B and a new value C, the gain is closed-form. Once B is established, the posterior odds of a real change are about h(1-eps_s) / ((1-h) eps_s). Latest-wins is therefore the special case eps_s = 0, and the gate is the Shiryaev-Roberts quickest-change detector. It is the text-memory analogue of TTT3R's closed-form, alignment-derived learning rate. (4) Retract, scope and evict: a retraction adds a negative likelihood on the named value. A temporary-scope observation opens a child sub-state that is evicted at expiry, which restores the parent value with no new evidence. Run-length hypotheses with mass below 1e-3 are pruned to bound memory. (5) Versions: superseded values are never deleted. They stay as time-stamped version siblings, so deterministic operators can answer previous, first and count queries. (6) Retrieve: the reader prompt gets the MAP value. When the posterior is split (max below 0.8), it also gets labelled alternatives with probabilities and a short evidence trail. (7) Theory, CPU only: treat Memini/Benna-Fusi coupled fast/slow edges as a linear filter. This gives an entrenchment law, update latency L of about a*R*exp(-lag/tau_slow) + b up to saturation of the slow variable, and the false-update rate P_f. In contrast, the Bayes detector's latency at a fixed P_f is bounded in R under a stationary hazard. (8) A reduced diagnostic covers only the R x lag x noise x scope axes that MINTEval, MemOps and CMI lack. It measures entrenchment slope and false-update rate for long context, BM25 and dense RAG, Mem0, A-Mem, Graphiti, Memini-style edges, per-session LoRA on Qwen2.5-1.5B, and our slots. It then tests whether these two fingerprints predict LongMemEval-KU and MemOps stale-value rate under leave-one-system-out.

**Datasets.**

- MemoryAgentBench FactConsolidation-SH and FactConsolidation-MH (Conflict Resolution), github.com/HUST-AI-HYZ/MemoryAgentBench, ICLR 2026
- LongMemEval-S, knowledge-update and temporal-reasoning categories, github.com/xiaowu0162/LongMemEval, ICLR 2025
- MemOps (github.com/MemTensor/MemOps, MIT): Update and Forget slices with confirmed, tentative and retracted labels and the recency_trap knob. Built on UltraChat.
- Implicit belief-staleness benchmark with BBT (github.com/abaabaaba-1/StaleMemory-anon): a deterministic generator, 1,200 canonical scenarios over 8 attributes, plus a 240-scenario held-out stress set
- LoCoMo (snap-research/locomo): mine about 200 natural corrections, hedges and temporary statements and verify them by hand as a natural-noise check
- ReBind-lite (ours, templated, CPU-generated). It covers only the axes others lack: reinforcement count R in {1,2,4,8,16,32}, lag up to 200 sessions, stance noise in {0, 0.1, 0.2, 0.4}, temporary scope with expiry, and retractions.
- Optional if data is released by the time we run: MINTEval (2605.18565), CMI-Travel (2608.07622), STALE (2605.06527)

**Models / checkpoints.**

- Qwen/Qwen2.5-7B-Instruct (extractor and reader; YaRN 128K for the long-context arm) served with vLLM
- Qwen/Qwen3-8B (second reader for robustness)
- Qwen/Qwen2.5-1.5B-Instruct (per-session LoRA SFT arm via PEFT)
- BAAI/bge-small-en-v1.5 (dense RAG retriever and entity/attribute canonicalization)
- gpt-4o (judge for the official LongMemEval eval script) and gpt-4o-mini (cross-check extractor), API only

**Baselines.**

- Full context / 128K long-context reader (Qwen2.5-7B-Instruct)
- BM25 RAG and dense RAG (bge-small) over raw sessions and turns
- Mem0 (mem0ai/mem0, pinned release, local Qwen backend)
- A-Mem (agiresearch/A-mem)
- Zep/Graphiti bi-temporal graph (getzep/graphiti)
- HippoRAG 2 (OSU-NLP-Group/HippoRAG)
- Deterministic latest-wins versioning with history operators (freshness recipe 2606.01435; tempora as the open implementation)
- MemTxn-style evidence-gated commits (2607.27834, reimplemented from the paper)
- BeliefMem Noisy-OR belief slots (2605.05583, reimplemented)
- BBT: Dirichlet-smoothed EMA plus 4-gate revision (StaleMemory-anon)
- Memini-style coupled fast/slow edge memory (2605.05097, reimplemented)
- Simple heuristics: k-confirmation rule and fixed-gain EMA
- Per-session LoRA SFT on Qwen2.5-1.5B (weights store)
- Ablations: gain=1 (latest-wins with the same extractor), one global hazard, stance ignored (constant eps), no temporary sub-states, no version siblings, MAP-only retrieval
- Oracle slot store (gold tuples) to separate extraction error from gating error

**Metrics.**

- Current-value accuracy on the clean, noisy (hedged/hearsay), tentative/retracted and temporary-scope subsets
- False-update rate (spurious overwrites); MemOps stale-value rate and recency-trap error
- Update latency: the number of reliable new-value observations before the system answers the new value. Entrenchment slope dL/dR and the lag dependence.
- Expiry-recovery accuracy (correct revert after a temporary scope ends)
- History accuracy: previous, first and count queries
- Slot calibration: ECE and Brier score; selective accuracy and AURC
- LongMemEval-S knowledge-update and temporal-reasoning accuracy (official judge); FactConsolidation-SH/MH exact match; BBT-benchmark detection and persistence
- Fit of the analytic entrenchment law to simulated and empirical latency (R^2, relative slope error)
- Predictive validity: leave-one-system-out Spearman rho between the (entrenchment slope, false-update rate) fingerprints and real benchmark subscores
- Cost: LLM calls and tokens per session, wall-clock per session

**First experiment (go/no-go).**

Week 1, CPU only. Implement the change-point slot filter, the Memini-style linear coupled fast/slow edge model, latest-wins, k-confirm, fixed-gain EMA and a BBT-like Dirichlet-EMA. Run them on tuple streams with no LLM, crossing R in {1..32}, lag in {1..200}, eps in {0, .1, .2, .4}, and temporary scope. Check that measured latency matches the predicted entrenchment slope within about 15%, and that the gate traces a better latency-vs-false-update Pareto front than all heuristics. Week 2, one 48-80 GB GPU, about 25 GPU-hours. Serve Qwen2.5-7B-Instruct with vLLM and extract tuples on: 300 FactConsolidation-SH items, the MemOps Update slice with recency_trap on (tentative and retracted), 300 implicit-staleness scenarios, and 200 ReBind-lite streams. Compare four arms: change-point slots with LLM extraction, change-point slots with oracle extraction, latest-wins slots from the same extractor (gain=1), and Mem0 (pinned) plus full context. GO if all three hold. (a) On the noisy, tentative and retracted subsets the gate beats gain=1 by at least 10 points and halves its false-update rate, while staying within 2 points on clean SH. (b) LLM extraction keeps at least 60% of the oracle gain. (c) Slot ECE is below 0.15.

**Kill criterion.**

Drop the method if any of these happen by the end of week 2. (1) With oracle extraction, the change-point gate does not beat both a tuned k-confirmation rule and latest-wins by at least 5 points on the noisy, tentative and retracted subsets. That would mean the Bayesian machinery adds nothing over a heuristic. (2) Qwen2.5-7B stance and scope extraction F1 is below 0.7, or LLM extraction loses more than 70% of the oracle gain. (3) Mem0, Graphiti or BBT already have under 5% false-update rate on MemOps recency_trap. If the method dies but the entrenchment law fits simulation and data (R^2 above 0.8), downgrade to a short theory-plus-diagnostic paper. If a full read of MINTEval or CMI shows they already report latency-vs-repetition-count across stores, cut the diagnostic entirely.

**Compute.** 120 GPU-hours over 14 weeks. 1x A100 80GB, or 1x L40S 48GB for all but the 128K long-context arm. CPU for all theory and filter simulations. About $150-300 of API calls for the gpt-4o LongMemEval judge and the gpt-4o-mini extraction cross-check.

**Risks.**

(1) Crowded space. The delta rests on the quickest-change framing, the entrenchment law, temporary scope and calibration. If MINTEval or CMI turn out to report latency vs repetition count, the diagnostic adds little. (2) Extraction may dominate. A 7B model may miss hedges, hearsay and temporary scope; the oracle-extraction arm and the gpt-4o-mini cross-check bound this. (3) Natural data is mostly clean updates, as in LongMemEval-KU and FactConsolidation, so headline gains may be confined to MemOps, BBT-bench and our grid. The LoCoMo-mined natural-noise set guards against reviewers seeing the variants as built for the method. (4) The noise rates and hazards need calibration data. Attribute classes with few changes rely on priors, and an empirical-Bayes split is required. (5) The slot schema covers attribute-like facts, not procedural or episodic memory, so we claim only that scope. (6) Memini, MemTxn and BeliefMem may lack public code, so reimplementation fidelity can be challenged; release the reimplementations and check them against reported numbers where possible. (7) Leave-one-system-out predictive validity has only about 10 systems; report it with bootstrap CIs as secondary. (8) The pinned versions of Mem0, Graphiti and A-Mem change quickly; freeze commits.

**Target venue.** ICML 2027 (theory plus method), with ACL 2027 via the ARR February cycle as a fallback. A workshop version of the entrenchment law is possible at an ICLR 2027 memory/continual-learning workshop.

**First-round prior-work verdict: partially_novel.** Repositioned per the novelty check: the method and theory lead, and the diagnostic is reduced to the entrenchment (R) and lag axes plus predictive validity. MINTEval, MemOps and CMI already cover cross-store interference benchmarking, recency traps and update suppression, so we make no 'first cross-store benchmark' claim. The nearest methods differ from ours. BeliefMem uses Noisy-OR slot beliefs with no change model or calibration. MemTxn uses evidence-gated commits. The freshness recipe and tempora use latest-wins with history operators. BBT (StaleMemory-anon, found in this step) uses a Dirichlet-smoothed EMA with 4 gates for implicit staleness, has no hazard, temporary scope or calibration, and must be a baseline. The defensible delta has three parts: casting overwrite as Bayesian quickest-change detection, with stance-dependent noise and attribute-class hazards, where latest-wins is the eps=0 special case; an analytic entrenchment law for Memini/Benna-Fusi coupled memories; and expiring temporary-scope sub-states with calibrated posteriors. Caveat: the web-search budget was exhausted and arXiv was unreachable. MINTEval, CMI, MemTxn and 2606.01435 were judged from snippets or the idea text only, so read them in full before week 1 ends.

**Closest work found in the first round:**

- [Controlled Memory Interference in Continual LLM Agents (CMI-Travel)](https://arxiv.org/abs/2608.07622) — arXiv 2608.07622, Aug 2026. A controlled interference diagnostic for agent memory. It varies memory scale and the relationships between accumulated memories while holding the query and target state fixed. It finds that repeated history and same-slot conflict strongly suppress adoption of valid updates. This is close to the idea's 'entrenchment slope vs R' and update-suppression claims. It separates interference that happens before the target is retrieved from interference after retrieval, and it uses CMI data to train interference-aware memory. Differences: it runs on MemoryArena travel planning, not AB-AC fact rebinding. From the search snippets, it does not compare context, RAG, consolidated memory and LoRA stores, and it has no change-point gate. I saw only search snippets, because arxiv was unreachable.
- [MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems](https://arxiv.org/pdf/2605.18565) — arXiv 2605.18565, May 2026. A benchmark built around interference from frequently updated facts, with 15.6k QA over contexts averaging 139k tokens. It covers state tracking, dialogue, Wikipedia revisions and Git commits, and has history-style subsets. It evaluates long-context LLMs, RAG and memory-agent frameworks, and a third-party listing says it forces handling of both proactive and retroactive interference. This partly pre-empts the cross-store interference evaluation. Uncertain: whether it reports separate PI and RI indices, fan or entrenchment per store. It does not appear to include LoRA or propose a method.
- [MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon Conversations](https://github.com/MemTensor/MemOps) — arXiv 2607.12893, Jul 2026. Operation-level diagnostics: Remember, Forget, Update, Reflect and trajectory operations. Each value carries a confirmed, tentative or retracted label, and a 'recency_trap' knob tests whether a later tentative or retracted value wrongly overwrites the confirmed state. Metrics include stale-value rate and over-forget or leakage, and StateTransition and trajectory probes cover old-to-new history. Systems are long context, RAG variants and Mem0/MemOS. This covers much of the idea's noisy, retraction and false-update variants and its history accuracy. It has no temporary-scope sub-states, no PI/RI/fan/entrenchment decomposition and no Bayesian method.
- [Belief Memory: Agent Memory Under Partial Observability (BeliefMem)](https://arxiv.org/pdf/2605.05583) — arXiv 2605.05583, May 2026. Stores candidate conclusions as probabilistic beliefs in structured semantic slots (subject, predicate, qualifiers) with evidence references and temporal info, and does not overwrite them. Each probability is updated by Noisy-OR as new observations arrive. This is close to the idea's per-slot discrete belief with alternatives returned when the posterior is split. It has no change-point or hazard model, and its confidences are LLM-extracted rather than calibrated.
- [Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics (Memini)](https://arxiv.org/pdf/2605.05097) — arXiv 2605.05097, May 2026. Graph memory in which each edge has coupled fast and slow variables (Benna-Fusi). Repeated co-occurrence consolidates an edge and weak edges decay. This is the system the idea's entrenchment law analyses. It does not derive update latency as a function of reinforcement count, and it does not add a change-point gate.
- [LLMs Remember First, Forget Last: Dual-Process Interference in LLMs](https://arxiv.org/pdf/2603.00270) — arXiv 2603.00270, 2026. An AB-AC paradigm inside the context window. It probes first-value vs latest-value recall with identical inputs, across 39 LLMs. It finds that proactive interference dominates, that PI and RI are uncorrelated, and that RI failures are retrieval misses while PI failures are early-item intrusions. The idea already acknowledges it. It also supplies the PI/RI/intrusion metric definitions the fingerprints would reuse.
- [Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates (DKI)](https://arxiv.org/html/2603.12271v1) — arXiv 2603.12271, Mar 2026. Treats successive AB to AC rebinding of one cue as retrieval competition. It finds an earliest-vs-latest accuracy gap that grows with the number of updates N. This covers the N axis of the AB-AC grid, but only in context.
- [Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs (PI-LLM)](https://arxiv.org/html/2506.08184v3) — arXiv 2506.08184, 2025. Streams key-value updates and queries only final values. Retrieval accuracy falls log-linearly with the number of interfering updates, even at constant input length. This is in context only.
- [Analyzing Memory Effects in LLMs through the Lens of Cognitive Psychology](https://arxiv.org/html/2509.17138v2) — arXiv 2509.17138, 2025. Tests fan-effect paradigms on LLMs and finds human-like difficulty with overlapping facts. This pre-empts 'fan effect in LLMs' in general, but not a per-store fan slope.
- [The Interference Gap: Comparing Retrieval Bounds in Human Memory and RAG Systems](https://arxiv.org/pdf/2606.28327) — arXiv 2606.28327, Jun 2026. Maps the fan effect onto a signal-detection framework for RAG retrieval. This partly pre-empts the claim that vector RAG is dominated by the fan effect, from the theory side. The empirical scope is unclear from the snippet.
- MemTxn and the deterministic freshness/versioning recipe (cited in the idea as 2607.27834 and 2606.01435) — arXiv 2026. The idea itself names these as baselines for evidence-gated commits and latest-wins versioning with history. I did not verify them in this search, so no URL is reported.

---

## 11. When Does Folding Recurrent States into Memory Work? Ridge-Optimal, Protected Consolidation of Session States in Linear-Attention LMs

*First-round title: Folding Sessions into Weights: When Can Linear-Attention Fast States Be Consolidated in Closed Form?*

**Judge:** overall 5/10, tier B. Scores out of 5: novelty 2, low compute 3, impact 4, speed of first signal 4.
It hits the fast state to weights consolidation that the doc names as the frontier, and it is gradient-free. The recheck, however, found PRECOG/SMC (Aug 2026), State Soup, Document Soupability, FAAST and the Sleep paper, so only the ridge/Gram-protected merge and the diagnostic map remain. It is best kept as a chapter in arc 3 rather than a lead project.

**Biggest weakness:** A per-head linear ridge fold is exact only for one layer. In deep multi-layer models the fresh-context states and queries drift, so the fold may not beat averaging or PRECOG composition. Fact recall from injected states in public 1.3-1.5B models may be too weak to measure.

**One line.** Keep a few small per-head statistics from each session. Merge many sessions' end-of-session recurrent states into one persistent initial state or read term using ridge regression that is recency-weighted and protected against generic text (RegMean/AlphaEdit applied to fast weights). Map when this gradient-free fold works across DeltaNet, GLA, Mamba2 and RWKV-7, and place it on the recall-vs-KL frontier against LoRA, gradient state tuning and RAG.

**Memory store.** cross-store: fast state -> slow parameters (a persistent consolidated initial state, or a non-decaying low-rank read term per head)

**Task.** Multi-session fact retention and knowledge updating in recurrent LMs of 1.3-1.5B. There are 10 to 100 short sessions of 200-400 tokens. Each states about 10 facts in natural text: (i) MQAR-style entity-attribute sentences with single-token values, (ii) bioS-style synthetic biographies, (iii) a short-answer subset of TOFU fictitious-author facts, and (iv) CounterFact counterfactual statements. Some later sessions contradict earlier ones. The test is a fresh session with no context that asks held-out paraphrase questions.

**Gap in the big picture.** The doc's open problem is consolidation: what moves from context into the weights, and when. This project tests one very cheap answer for one store pair, fast state to slow parameters. It also targets the doc's fast-state failure mode ('lost when the session ends unless consolidated') and its lesson that forgetting tracks how far an update moves the model (sparse memory finetuning 2510.15103; RL's Razor). The fold explicitly bounds the shift on generic inputs, so that lesson can be tested without gradients. Because the fold uses per-session sufficient statistics instead of raw text, it also addresses the data-buffer failure mode in the doc (storage and privacy).

**Hypothesis.** On RWKV-7-1.5B and DeltaNet-1.3B, a 20-session fold that is recency-weighted, ridge-solved over query Grams and null-space-protected will recover at least 50% of mean per-session in-context fact recall (exact match) in a fresh context, while raising generic WikiText-103 next-token KL by less than 0.02 nats/token. On the recall-vs-KL frontier it will beat naive state summation, sequential state carry-over and LoRA SFT on session text.

**Method.**

(1) Write. Each recurrent head h has an end-of-session state S_i (d_v x d_k), read as o = S q. While session i runs, we also accumulate small sufficient statistics per head: the self-read query Gram C_i = sum_t q_t q_t^T, plus, for delta-rule models, the moments A_i = sum_t beta_t v_t k_t^T and G_i = sum_t beta_t k_t k_t^T. That is 2-3 matrices of 128x128 per head, and no text is kept. (2) Consolidate. Per head, solve M = (sum_i w_i S_i C_i)(sum_i w_i C_i + lambda C_ref + mu I)^-1. This is RegMean-style merging applied to fast-weight states. C_ref is the query Gram on about 1M generic FineWeb-Edu tokens, so the term forces M q ≈ 0, which is fresh-session behaviour on generic queries (AlphaEdit-style protection). w_i = rho^(N-i) is a recency weight that resolves contradictions. Written as recursive least squares with a forgetting factor, each new session costs one d x d solve per head. A 're-solve' variant uses (A_i, G_i) to compute the exact cross-session ridge solution that the delta rule only approximates online (a MesaNet-style offline 'sleep'). (3) Layers are folded bottom-up. At layer l, probe-query Grams are recomputed in a fresh context with layers below l already folded, as MEMIT does layer by layer. This handles the main risk that test-time queries do not align with the keys written during the session. (4) Deploy, read and route. Variant (a) injects M as the persistent initial state of every new session, with zero conv state; the model's own gates and delta erasure then decay or overwrite it. Variant (b) adds a non-decaying read term o += gamma M q, a closed-form adapter of rank <= d_k. Only the top 10-30% of heads are folded, ranked by a cheap per-head injection-gain score on a dev split. (5) Analysis, the main claimed contribution. Predict per-head fold success from decay half-life, the effective rank of S_i, and the alignment between fresh-context queries and session keys. This gives a map of where folding works and where it fails across no-decay DeltaNet, Hebbian GLA/Mamba2 and decaying generalized-delta RWKV-7. Report every method on the recall-vs-generic-KL frontier.

**Datasets.**

- Synthetic MQAR-style natural-language key-value sessions, generated on CPU: fictional entities with single-token attributes, separate templates for writing and testing
- Synthetic bioS-style biographies (Physics of LMs 3.1 format, regenerated with fresh name and attribute pools), including contradiction sessions
- TOFU (locuslab/TOFU), a short-answer subset of fictitious-author QA
- CounterFact (ROME/MEMIT release), counterfactual statements as sessions, for comparison with MLP editing
- WikiText-103 validation (Salesforce/wikitext) for KL and perplexity drift
- FineWeb-Edu sample (HuggingFaceFW/fineweb-edu, sample-10BT), about 1M tokens for the generic reference query Gram C_ref
- lm-eval-harness zero-shot: LAMBADA, PIQA, HellaSwag, ARC-e, ARC-c, WinoGrande

**Models / checkpoints.**

- fla-hub/rwkv7-1.5B-world (RWKV-7, generalized delta rule with decay; the HF name is from memory, and the official BlinkDL RWKV-7 g1 1.5B weights are the fallback)
- fla-hub/delta_net-1.3B-100B (DeltaNet, no decay; closest to Chen et al.'s exact-conversion setting; name from memory)
- fla-hub/gla-1.3B-100B (gated linear attention, Hebbian with decay; name verified in the flash-linear-attention README)
- state-spaces/mamba2-1.3b (Mamba2, scalar decay, Hebbian-style)
- Gated DeltaNet 1.3B from NVlabs/GatedDeltaNet, only if a public checkpoint exists (unverified)
- Library: flash-linear-attention 0.5.2 (PyPI, verified), with recurrent-state cache injection via its Cache object

**Baselines.**

- No memory (fresh session)
- In-context oracle: the target session plus the question (the per-session ceiling)
- Sequential state carry-over through all sessions (identical to concatenated long context for a pure recurrent LM)
- Last-session-only state carry-over
- Naive state sum or mean, i.e. Chen et al.'s single-context conversion applied per session and summed
- RegMean-style fold with key Gram instead of query Gram, and without protection or recency (ablations)
- BM25 RAG over session texts, top-k sessions placed in context
- LoRA SFT (rank 8/16 on q/k/v/o) on session text, joint and sequential
- Gradient-trained persistent initial state (RWKV state-tuning style, Cartridges-like): the same parameter learned by gradient, to isolate closed form vs gradient
- AlphaEdit/MEMIT MLP edits on the CounterFact track only (they need explicit triples; this compares MLP and fast-weight stores)

**Metrics.**

- Fresh-session fact recall: exact match on first token and on the greedy-decoded answer, using held-out paraphrase templates
- Retention by session age (backward transfer across sessions) and fraction of in-context recall recovered
- Contradiction accuracy (latest value) and stale-value rate
- Generic drift: next-token KL(base || folded) on WikiText-103 validation, change in perplexity, change in lm-eval zero-shot average
- Recall-vs-KL frontier (area under it), following the sparse memory finetuning comparison
- Cost: consolidation wall-clock, bytes stored per session (statistics vs text vs LoRA), inference overhead
- Diagnostics per head: decay half-life, effective rank of S_i, fresh-query/session-key alignment, injection gain

**First experiment (go/no-go).**

Weeks 1-2, one 24-48 GB GPU, about 10 GPU-hours. Day 0: re-run the novelty search (see novelty_note). Days 1-3: load RWKV-7-1.5B and DeltaNet-1.3B through fla. Write hooks to read per-layer, per-head end states and query Grams, and to inject a recurrent_state as the initial cache. Sanity check: injecting one session's full state and then asking the question must reproduce in-context recall to numerical precision. Days 4-10: generate 20 MQAR-style sessions (10 single-token-value facts each, 200-300 tokens with filler; 1,000 test questions in held-out templates). For N in {1, 2, 5, 10, 20}, compare: no memory, in-context oracle, last-session carry-over, sequential carry-over, naive sum, and ridge fold with self-query Gram (sweep lambda over 5 values, rho in {1, 0.9}, head fraction in {10, 30, 100}%; variants (a) and (b)). Measure KL on 200 WikiText-103 validation sequences. GO if, at N=10, the ridge fold recovers at least 50% of mean per-session in-context exact match, beats naive sum by at least 10 points, beats sequential carry-over on sessions older than 5, and keeps generic KL at or below 0.02 nats/token. NO-GO if the fold at N=5 scores below 2x no-memory exact match or below naive sum on both models.

**Kill criterion.**

Drop the method claim and keep only the diagnostic paper (which heads in public 1.3B recurrent LMs hold fresh-context-retrievable facts, and how far folding moves generic KL) in either of two cases: (1) on both RWKV-7-1.5B and DeltaNet-1.3B, the best closed-form fold of 10 sessions recovers less than 30% of mean per-session in-context recall at generic KL at or below 0.02 nats/token; (2) at equal or lower compute, gradient-trained initial-state tuning beats the fold on the recall-vs-KL frontier. Drop the project entirely if the Day-0 novelty re-check finds a published gradient-free, multi-session state-to-weight merge for linear-attention or SSM LMs that already includes protection and contradiction handling.

**Compute.** 120 GPU-hours over 12 weeks. 1x A100-80GB or L40S-48GB (an RTX 4090 24GB is enough for all inference and for LoRA on 1.5B models). Consolidation is CPU/GPU d x d solves, taking seconds.

**Risks.**

(1) Misalignment between test-time queries and session keys. Without the session in context, lower layers compute different features, so questions may not produce queries that hit the stored keys. Bottom-up refitting and probe-query Grams address this, and the analysis measures it. (2) The non-decaying variant (b) may push activations out of distribution. Variant (a), which uses the model's own gates, is the safer default. (3) Facts in small recurrent LMs may live mainly in MLPs or short-conv paths, not in recurrent heads, which caps recall. The CounterFact comparison with AlphaEdit makes this measurable. (4) fla-hub 1.3B/100B-token models have weak in-context recall, which lowers the ceiling. RWKV-7 (trained on trillions of tokens) is the main model for this reason. (5) Multi-token answers need coordinated reads across positions, so first-token exact match is reported alongside greedy-decode exact match. (6) Mamba2's channel-wise discretisation and Hebbian updates make the 're-solve' variant out of distribution there, so it is evaluated with the S_i-preserving fold only. (7) BM25 RAG will likely win on raw recall. The claim is made on stored bytes, zero context cost and the generic-KL frontier, not on recall alone. (8) Novelty is unverified because search was unavailable. A close 2026 paper could reduce this to the diagnostic contribution.

**Target venue.** ICML 2027 (deadline around late January 2027). Fallback: COLM 2027, or an ICLR 2027 workshop on memory and continual learning for the diagnostic version.

**First-round prior-work verdict: partially_novel.** The building blocks all exist. Chen et al. (ICML 2024) give the exact conversion of a context into weights. The merge formula is RegMean-style regression merging plus AlphaEdit-style null-space protection. MesaNet-style ridge-optimal fast weights, RWKV state tuning, Cartridges and 'Language Models Need Sleep' (gradient-based) all move context into persistent parameters, and FAAST builds closed-form fast weights. The defensible delta is therefore (a) gradient-free consolidation of a model's own end-of-session fast states across many sessions, from per-session sufficient statistics, with contradiction handling by recency-weighted RLS and bottom-up fresh-query refitting; and (b) the analysis of when folding works or fails across decay and update rules, plus the comparison on the recall-vs-KL frontier. The paper is positioned as an analysis plus a method, not as the conversion identity. Caveat: the shared web-search budget was used up in this run too, and arXiv/HF fetches failed. No closest work could be opened. The URLs for Chen et al., AlphaEdit (2410.02355) and Cartridges (2506.06266) are from memory, and RegMean, MesaNet, RWKV state tuning and State-offset Tuning are further unverified overlaps that the Day-0 re-check must clear.

**Closest work found in the first round:**

- [Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers (Chen et al.)](https://arxiv.org/abs/2406.02847) — ICML 2024. NOT VERIFIED IN THIS RUN. The idea itself cites this paper, and the URL is my own guess at its arXiv ID; the fetch failed and I could not check it. As the idea describes it, the paper already shows the central identity: a linear-attention context state S used as o = S q can be written exactly as a weight term. That covers the 'a session state is already a weight edit' step for one context without decay. What it reportedly does not cover is merging many sessions, decaying delta-rule models, null-space protection and head selection. All of this comes from the idea's text.
- [Language Models Need Sleep (arXiv 2605.26099, as cited in the idea)](https://arxiv.org/abs/2605.26099) — arXiv 2026. NOT VERIFIED. The arXiv fetch failed with a proxy 403. According to the idea, it learns an offline consolidation of context or fast memory into weights and needs training. If it consolidates recurrent fast states across sessions, it is the closest threat on the problem itself, though the idea says it uses gradients, not a closed form.
- [FAAST (arXiv 2605.04651, as cited in the idea)](https://arxiv.org/abs/2605.04651) — arXiv 2026. NOT VERIFIED (fetch failed). According to the idea, it builds closed-form fast weights from labeled examples, so the closed-form fast-weight construction is already published. The idea's claimed difference is that it uses the model's own end-of-session states and merges across many sessions. I could not check how far FAAST goes.
- AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models — ICLR 2025. NOT VERIFIED IN THIS RUN. The idea names it, and I have no verified URL. The null-space or ridge protection against keys from generic text is AlphaEdit's mechanism, and the least-squares edit with a preservation term is MEMIT-style. The idea applies these to linear-attention read paths instead of MLPs, and uses session states as targets instead of explicit triples.
- Cartridges (trained KV cache / context compression by gradient) — 2025. NOT VERIFIED. The idea names it, and I have no verified URL. It turns context into a persistent learned artifact by gradient descent. That is the same goal (keep session content without context) with a different mechanism (gradient-trained KV prefix, not a closed-form weight fold).

### Second-round re-check (live search)

**Revised one line.** State souping and PRECOG-style state injection already persist sessions by averaging recurrent states. We replace averaging with a gradient-free, per-head ridge fold, built on query Grams, protected against generic text and weighted toward recent sessions. We then map when it beats averaging, gradient state tuning, LoRA and RAG on the fact-recall vs generic-KL frontier across RWKV-7, DeltaNet, GLA and Mamba2.

**Verdict: partially_novel; recommendation: pursue_with_pivot.**

**What is taken and what is still new.** I tried to refute the plan's novelty with 12 searches. These covered its cited IDs, state merging and soups, closed-form fast weights, state injection, gradient-free consolidation, and knowledge editing in SSMs. I found no published gradient-free, multi-session merge that turns an LM's own recurrent states into a persistent state or weight term and also includes ridge/Gram weighting, generic-text protection and recency-based contradiction handling. The plan's Day-0 drop condition is therefore not triggered.

The space around it is more crowded than the plan assumes, though, and two of its implicit claims are taken.
(1) 'Persist session states by injecting them as the initial state' is already published. State Soup (2406.08423) does linear state interpolation. Document soupability (2505.24033) pools states. Most directly, PRECOG/SMC (2608.02560, Aug 2026) composes top-k stored SSM states by softmax weighting into h_init on a 1.2B gated SSM and 'consolidates short-term episodic states into long-term semantic memory'.
(2) 'Closed-form least-squares fast weights instead of gradients' is FAAST (2605.04651).
Consolidating context into SSM fast weights during a sleep phase is Lee et al. (2605.26099, ICML 2026), but it uses a learned rule trained by backprop.

The remaining, defensible delta has four parts:
(a) The merge rule is principled rather than heuristic. A per-head, RegMean-style ridge solve over self-read query Grams, with a generic-reference Gram for protection and recency-weighted RLS, replaces averaging or softmax composition. The test is that it beats State-Soup averaging and PRECOG-style top-k composition as N grows, where PRECOG itself reports degradation.
(b) Bottom-up refitting of fresh-context queries handles query/key misalignment.
(c) The method runs on frozen public decaying delta-rule LMs (RWKV-7, DeltaNet, GLA, Mamba2) with no training at all, unlike Sleep.
(d) The analysis maps where folding succeeds or fails by decay, rank and alignment, on a recall-vs-generic-KL frontier against state tuning and S0 tuning, LoRA, ROME/AlphaEdit and RAG.
The claimed 'exact conversion' and 'state as persistent memory' framing has to go.

**Required revisions.** 1) Fix the closest-work list. Chen et al. is verified as arXiv 2406.02847 (ICML 2024). 'Language Models Need Sleep' 2605.26099 is Lee, McLeish, Goldstein and Fanti (ICML 2026; v3 is titled 'Do Language Models Need Sleep?'). It uses a learned local rule with backprop through offline SSM passes. Also note the same-title Behrouz et al. paper (2606.03979, not verified on arXiv). FAAST 2605.04651 is verified as closed-form min-norm least-squares fast weights. Cartridges 2506.06266 (ICLR 2026) and AlphaEdit 2410.02355 are verified.

2) Add the missed close works and cite them prominently: PRECOG/SMC (2608.02560, persistent SSM state injection plus episodic-to-semantic state consolidation on a 1.2B SSM), State Soup (2406.08423) and Document Soupability (2505.24033).

3) Relabel the 'naive state sum/mean' baseline as State Soup averaging, and add PRECOG-style top-k softmax-weighted state composition as a baseline. This becomes the main comparison: the ridge fold must win as N grows, because averaging degrades there.

4) Add an S0-tuning / RWKV state-tuning gradient initial state (already partly present) and a FAAST-style min-norm closed-form fit without protection or recency, as an ablation separating 'closed form' from 'protected + recency'.

5) On the CounterFact track, add ROME on SSMs (Locating and Editing Factual Associations in Mamba, 2404.03646) next to AlphaEdit and MEMIT.

6) Drop or soften three claims: (i) 'first to persist session states as an injected initial state'; (ii) 'closed-form fast weights' as a novelty in itself; (iii) 'AlphaEdit-style null-space'. The plan's term is soft ridge protection, so describe it that way.

7) Make the headline contributions these: the RegMean-on-fast-states merge with reference-Gram protection and recency RLS; bottom-up fresh-query refitting; and the diagnostic map (decay half-life, state rank, query/key alignment) across four architectures on the recall-vs-generic-KL frontier.

8) The kill criterion should now also trigger if PRECOG-style composition matches the ridge fold within noise at N >= 10. In that case, fall back to the diagnostic paper.

The compute budget (about 120 GPU-hours) is unchanged.

**Baselines to add.**

- State Soup linear state interpolation/averaging (Pióro et al., 2406.08423), relabelling the plan's naive-sum baseline
- PRECOG-style top-k softmax-weighted state composition injected as h_init (2608.02560)
- Document-soup state pooling for Mamba2 (2505.24033), per-query pooled state
- S0 tuning / RWKV-7 state tuning (2504.05097): gradient-learned persistent initial state on the same session data
- FAAST-style min-norm least-squares closed-form fast weights without protection or recency (2605.04651), as an ablation
- ROME rank-one editing on Mamba (2404.03646) on the CounterFact track

**Closest work (second round):**

- [Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers (Chen, Hu, Jin, Lee, Kawaguchi)](https://arxiv.org/html/2406.02847v2) — ICML 2024 (PMLR 235). VERIFIED: arXiv 2406.02847 is this paper. Its ICLCA algorithm moves the effect of in-context demonstrations exactly into bias terms of linearized transformers, and approximately for softmax/GPT-2. It is the single-context, no-merge identity that the plan builds on. It does not merge many sessions, handle decay or delta-rule models, protect generic queries, or resolve contradictions.
- [Structured Memory for Edge Language Models: Persistent Context and Corpus Retrieval via O(1) SSM State Injection (PRECOG + SMC; Madan Gopal et al., BrainChip)](https://arxiv.org/pdf/2608.02560) — arXiv Aug 2026 (2608.02560). CLOSEST NEW THREAT; the plan missed it. It works on a 1.2B gated SSM (TENNs-LLM). Corpora are pre-encoded as SSM hidden states, and at query time the top-k states are composed by softmax-weighted averaging into h_init and injected as the initial recurrent state. Its SMC layer 'consolidates short-term episodic states into long-term semantic memory' and fuses them with retrieved states. That is the plan's deploy variant (a), a persistent injected initial state built from session states, already published for SSMs. Per the snippets, the composition is a heuristic: the exact guarantee covers only top-1 single-chunk injection, and quality degrades as k grows. There is no ridge or query-Gram solve, no generic-text protection, no recency-weighted contradiction handling, and no generic-KL frontier.
- [State Soup: In-Context Skill Learning, Retrieval and Mixing (Pióro, Wołczyk, Pascanu, von Oswald, Sacramento)](https://arxiv.org/pdf/2406.08423) — ICML 2024 NGSM workshop (arXiv 2406.08423). It stores, retrieves and linearly interpolates Mamba-2.8B recurrent states as task vectors. Interpolation is exact for a single gated-linear layer and approximate in deep models. This is exactly the plan's 'naive state sum/mean' baseline, which is therefore prior work, not a strawman. It does not use ridge or Gram-weighted merging, protection, or fact retention across sessions.
- [Studying the Soupability of Documents in State Space Models (Jafari, Wang, Bergen, Berg-Kirkpatrick)](https://arxiv.org/html/2505.24033v2) — arXiv 2025 (2505.24033). It encodes documents independently and pools their Mamba2 states (for example by averaging) into one context state for multi-hop QA. The pooling needs finetuning to work well. This is multi-context state merging into one state, but it is per-query and not a persistent consolidated memory, and the merge is not closed-form regression.
- [Language Models Need Sleep (Lee, McLeish, Goldstein, Fanti); v3 titled 'Do Language Models Need Sleep?'](https://icml.cc/virtual/2026/74224) — ICML 2026 (arXiv 2605.26099). VERIFIED ID. The model periodically converts recent context into persistent fast weights in its SSM blocks by N offline recurrent passes with a learned local rule, trained end to end by backprop through sleep, before clearing the KV cache. The goal is the same, consolidating context into persistent SSM weights. The rule is learned and needs training of the architecture, and it is tested on synthetic and math tasks, not with frozen public LMs. A separate paper with the same title by Behrouz et al. (2606.03979, seen only via a document-sharing site) uses distillation plus RL 'dreaming', which is also gradient-based.
- [FAAST: Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation (Bao et al.)](https://arxiv.org/pdf/2605.04651) — arXiv May 2026 (2605.04651). VERIFIED ID. It compiles labeled examples into fast weights analytically in one forward pass, as the minimum-norm least-squares (pseudoinverse) solution, giving O(1) inference with no context. The 'closed-form least-squares fast weights instead of gradients' claim is therefore taken. FAAST does not use a pretrained recurrent LM's own end-of-session states, does not merge across sessions with recency, and does not protect generic text.
- [State Tuning: State-based Test-Time Scaling on RWKV-7; and S0 tuning on Qwen3.5 GatedDeltaNet hybrids](https://arxiv.org/html/2504.05097v1) — arXiv 2025 (2504.05097) / 2026. These learn a persistent initial state by gradient with frozen weights. They are the gradient counterpart of the plan's fold, and the plan's kill criterion (2) already names this baseline. One source argues that tuned states mainly elicit existing knowledge rather than inject new facts, which supports the plan's fact-injection framing as an open question. The S0 tuning snippet came without a verified arXiv ID.
- [Dataless Knowledge Fusion by Merging Weights of Language Models (RegMean; Jin et al.)](https://arxiv.org/pdf/2212.09849) — ICLR 2023 (arXiv 2212.09849). The merge formula M = (sum S_i C_i)(sum C_i)^-1 is RegMean's per-linear-layer closed form with input Gram matrices. The plan's novelty is applying it to fast-weight states, adding a protection term and recency, and refitting bottom-up. No search result applies RegMean to recurrent states.
- [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/pdf/2410.02355) — ICLR 2025 (arXiv 2410.02355). VERIFIED ID. Its null-space projection of edits against preserved-knowledge keys is the protection mechanism the plan uses via the lambda*C_ref term. The plan should call its version ridge-regularized soft protection, not exact null-space projection.
- [Locating and Editing Factual Associations in Mamba](https://arxiv.org/html/2404.03646v2) — COLM 2024 (arXiv 2404.03646). ROME-style rank-one editing inserts facts in Mamba-2.8B, so closed-form fact insertion into an SSM's MLP and projection weights exists. It should be a baseline on the CounterFact track in addition to AlphaEdit.
- [Cartridges: Lightweight and general-purpose long context representations via self-study (Eyuboglu, Ré et al.)](https://arxiv.org/abs/2506.06266) — ICLR 2026 (arXiv 2506.06266). VERIFIED ID. It trains a KV-cache artifact per corpus offline by context distillation on self-generated conversations, and Cartridges compose at inference. The goal is the same, but the method is gradient-based and transformer-only.

**Queries run:** Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers; "Language Models Need Sleep" arXiv 2026 consolidation; FAAST closed-form fast weights arXiv 2026; merging recurrent states linear attention SSM "state soup" OR "state merging" Mamba multiple contexts; [extended] consolidate fast weights into slow weights closed-form DeltaNet RWKV-7 multi-session memory without gradients; Fast Weight Attention for Continual Learning arXiv 2608.27763; persistent initial state recurrent language model knowledge injection state tuning RWKV Mamba "initial state" facts 2025; knowledge editing linear attention recurrent state model closed-form least squares DeltaNet Mamba editing facts null space; gradient-free context-to-weights consolidation continual learning across sessions LLM 2026 arXiv closed-form memory compilation; "Structured Memory for Edge Language Models" SSM state injection persistent context; PRECOG selective SSM retrieved states composed initial recurrent state injection preprint; Cartridges self-study arXiv 2506.06266 OR RegMean "Dataless Knowledge Fusion by Merging Weights" linear attention states

---

## 12. Revisits Break Persistence of Importance: An Oracle-Audited, Revisit-Stratified Study of KV-Cache Eviction in Streaming 3D Transformers

**Judge:** overall 5/10, tier C. Scores out of 5: novelty 3, low compute 3, impact 2, speed of first signal 4.
A closed-loop future-attention oracle combined with revisit stratification is a clean audit idea, and it is in the user's 3D area. But KV eviction for streaming 3D transformers is crowded (Evict3R, InfiniteVGGT, RegVGGT, STAC, RetrieveVGGT, OVGGT). The output is an audit of others' heuristics rather than a consolidation result.

**Biggest weakness:** The full-cache StreamVGGT at hundreds of frames is memory-heavy, and the oracle needs multiple passes. STAC or RetrieveVGGT may already report revisit behaviour. Impact on the doc's cross-store consolidation problem is low.

**One line.** Measure how far published KV-cache eviction policies for StreamVGGT and STream3R fall from a closed-loop future-attention oracle, separately on novel frames and on short-gap and long-gap revisits. Then test whether the minimal continual-learning fix, a place-quota buffer that treats places as classes, closes the long-gap part of that gap.

**Memory store.** The context/external store: the in-model global-attention KV cache of causal streaming 3D transformers (StreamVGGT, STream3R). Recurrent fast-state models (CUT3R, TTT3R) serve as fixed-memory reference points.

**Task.** Memory-bounded streaming 3D reconstruction (per-frame depth, camera pose, point maps) from indoor RGB video containing long-gap revisits of earlier places. The setting is a hard total token budget with no CPU offload. The work is inference-only.

**Gap in the big picture.** The anchor is lesson 3 of the big-picture doc: external memory moves the stability-plasticity dilemma instead of escaping it. With a finite context, old and new memories compete at retrieval (arXiv 2604.27003). In the doc's streaming-3D map, StreamVGGT's KV cache is that finite context, and it grows by about 100 MB per frame.

A crowded line of bounded-cache methods exists: Evict3R, InfiniteVGGT, OVGGT, GHOST, StreamCacheVGGT, FrameVGGT, RegVGGT, STAC and RetrieveVGGT. They rank tokens by attention history, redundancy, saliency or geometry, and they report sequence averages. None of them measures distance from an optimal eviction schedule. None tests the assumption that past or initial importance predicts future importance, an assumption RegVGGT states outright, in the case where it should fail: a revisit to a place that has been out of view for longer than the budget can hold. None reports results stratified by revisit gap.

LLM work has future-attention oracles (ForesightKV's Golden Eviction), but only for text. The continual-learning lesson that class-balanced buffers protect old classes under an unknown future query distribution (GDumb, class-balanced reservoir) has not been tested against an oracle in 3D. RetrieveVGGT and STAC have place-aware heuristics, but those are not measured against any optimum.

**Hypothesis.** At a 25% KV budget on StreamVGGT, attention- and redundancy-based eviction (Evict3R, InfiniteVGGT, RegVGGT-style) will be at least 1.5x worse than the full cache in depth AbsRel or RPE on long-gap revisit frames while staying within 10% on novel frames. A closed-loop future-attention oracle under the same budget will recover at least 50% of that loss. Cumulative or initial attention scores will predict the oracle's keep decisions for later-revisited tokens with AUROC <= 0.65, against >= 0.8 for other tokens.

**Method.**

(1) Oracle audit. Run full-cache StreamVGGT and STream3R on clips of <=500 processed frames (518x392 input gives about 1041 tokens per frame: 24 global layers, 16 heads, dim 1024). For each layer l, cached token i and later frame t', log the head-averaged attention mass a_l(i,t') in fp16, sparsified to mass >1e-5. The future utility of token i at time t is U_l(i,t) = sum over t'>t of a_l(i,t'); a variant weights each term by ||v_i|| to correct for attention not being importance. Under per-layer budget B, the open-loop oracle keeps the top-B tokens by U, and a classic next-use Belady variant evicts the token whose next above-threshold use is furthest away. The closed-loop oracle re-logs attention under its own evictions and repeats 2-3 times until the kept set stabilises; it is reported as an empirical reference, not a proven bound. The camera-head cache (one token per frame) and frame-0, camera and register tokens are never evicted by any policy.

(2) Persistence-of-importance test. Each online score is evaluated by AUROC/AP against the oracle's keep/evict label, stratified by whether token i's GT surface is re-observed later at a short or long gap. The scores are recency, cumulative attention (H2O/Evict3R), decayed attention (STAC), initial saliency (RegVGGT), key redundancy (InfiniteVGGT), FFN-residual magnitude (OVGGT), depth confidence and pose change (GHOST), and place rarity.

(3) Revisit-stratified protocol. Frames are labelled on CPU from GT depth and poses, using a depth-consistent reprojection overlap. A frame is novel if its overlap with every earlier frame is <0.3. It is a short-gap revisit if its overlap is >=0.3 with a frame inside the last W processed frames, where W is the budget in frame-equivalents, so long-gap means 'beyond what a sliding window of the same budget can hold'. It is a long-gap revisit if its overlap is >=0.3 only with frames older than W. A revisit consistency error (RCE) is also computed: the Sim3-aligned distance between predicted 3D points of GT-corresponding pixels at the first visit and at the revisit.

(4) Minimal CL fix, the place-quota wrapper (PQ). Each token gets a place cell c = voxel(predicted 3D point, edge = 1/16 of median predicted depth, which avoids scale ambiguity) x one of 6 cube-face viewing-direction bins. When the cache is over budget, the wrapper evicts from the cell with the most tokens, uses any base scorer (Evict3R, InfiniteVGGT, OVGGT, GHOST) to pick the token within that cell, and keeps a minimum quota q_min per visited cell, coarsening voxels if cells x q_min > B. Reservoir sampling over frames is the classic CL buffer baseline.

(5) A short analytic result motivates PQ. Assume the revisited place is unknown with prior p_c and per-place utility is concave, u(n) = 1 - exp(-n/kappa). The expected-utility-optimal allocation is then water-filling, n_c = kappa * max(0, log(p_c/lambda)). This reduces to equal quotas under a uniform prior. Attention-history scorers implicitly set p_c to recent attention, which is near zero for out-of-view places. kappa is fitted from the oracle logs.

**Datasets.**

- ScanNet v2 (val/test sensor streams): about 15 curated loop clips of 300-500 processed frames, subsampled at stride 2-5 from 1000-5000-frame scenes, with GT depth and poses
- TUM RGB-D: fr3/long_office_household (2585 frames, closes a loop), fr1/room and fr2/desk, subsampled to <=500 processed frames
- 7-Scenes: stitched A-B-A streams (seq-01 of scene A, a sequence of scene B, seq-02 of scene A; sequences of one scene share a world frame), a CL-style task-return stress test
- Neural RGB-D (NRGBD) synthetic scenes for Acc/Comp/NC point-cloud evaluation (in the StreamVGGT/STream3R eval pipelines; preprocessed sets at HF yslan/pointmap_regression_evalsets)
- KITTI odometry 00 and 05 (loop sequences), stretch goal only
- Released artifact: the revisit labels (novel / short-gap / long-gap per frame), the clip lists and the logged oracle schedules, as a small revisit-stratified benchmark

**Models / checkpoints.**

- lch01/StreamVGGT (Hugging Face; code github.com/wzzheng/StreamVGGT, verified; VGGT-1B-based causal aggregator: 24 layers, dim 1024, 16 heads, patch 14, 4 register tokens)
- yslan/STream3R (Hugging Face; code github.com/NIRVANALAN/STream3R, verified; StreamSession API with KV-cache management for causal and window modes)
- facebook/VGGT-1B (offline full-attention reference on <=200-frame clips, optional)
- CUT3R official checkpoint (github.com/CUT3R/CUT3R, verified) as the recurrent fixed-state reference
- TTT3R (github.com/Inception3D/TTT3R, verified; training-free on CUT3R) as the fast-weight state reference
- Point3R (github.com/YkiWu/Point3R, verified): pointer-memory appendix, stretch only

**Baselines.**

- Full cache (reference, <=500 processed frames on 80 GB)
- Sliding window plus frame-0 anchor (StreamingLLM-style sink)
- Uniform keyframe subsampling
- Random token eviction
- Reservoir sampling over frames (ER-style CL buffer)
- Evict3R (cumulative attention importance)
- InfiniteVGGT (key redundancy/diversity)
- OVGGT (FFN-residual scoring plus view-overlap historical anchors)
- GHOST (geometry-hierarchical eviction from pose change, depth-gradient variance and confidence)
- RegVGGT (initial-saliency regulated memory), the explicit persistence-of-importance hypothesis
- STAC (voxel-binned long-term spatial cache with merging), equal total budget, no offload
- RetrieveVGGT (pose-grid thinning plus query-key retrieval), equal total budget, no offload
- H2O / SnapKV-style LLM heuristics transplanted to the global-attention cache
- CUT3R and TTT3R (fixed-size recurrent state) as cross-store reference rows
- Open-loop and closed-loop future-attention oracle (ForesightKV-style Golden Eviction adapted to 3D)
- For methods without released code, faithful '-lite' reimplementations of the scoring rule only, disclosed as such

**Metrics.**

- Per-stratum (novel / short-gap / long-gap revisit) per-frame depth AbsRel and delta<1.25 (per-sequence median scale)
- ATE RMSE (Sim3-aligned), and RPE translation and rotation, also computed separately on revisit segments
- Acc / Comp / NC of fused point clouds (7-Scenes, NRGBD)
- Revisit consistency error (RCE): predicted-point distance between GT-corresponding pixels at the first visit and at the revisit
- Fraction of oracle gap closed = (E_base - E_policy) / (E_base - E_oracle), with base = sliding window, per stratum
- AUROC / AP of each online score against the oracle keep label, split by later-revisited vs other tokens
- Per-place retained-token distribution (Gini coefficient) and fitted kappa of the concave utility
- Peak GPU memory, FPS, and budget fraction (10%, 25%, 50%)

**First experiment (go/no-go).**

Weeks 1-2, one 80 GB GPU, about 15 GPU-hours.

Days 1-2:
- Install StreamVGGT (lch01/StreamVGGT) and wrap the aggregator's global-attention past_key_values with a token-mask eviction interface. Leave the camera-head cache intact.
- Write the CPU labeller that computes GT depth-consistent overlap and assigns each frame to novel, short-gap or long-gap.
- Select 6 ScanNet loop clips (about 400 processed frames at stride 3), TUM fr3/long_office_household (about 400 frames at stride 6) and one stitched 7-Scenes chess-fire-chess stream.

Days 3-5:
- Run the full cache with sparsified attention logging (8 clips, about 10 min each).
- At a 25% budget, run sliding window plus anchor, random, Evict3R-style cumulative attention, InfiniteVGGT-style key diversity, and the open-loop oracle.

Days 6-8:
- Compute per-stratum AbsRel, RPE and RCE.
- Compute AUROC of the online scores against oracle keep labels, split by later-revisited tokens.

Days 9-10:
- Run one closed-loop oracle iteration.
- Read the full STAC, RetrieveVGGT and RegVGGT PDFs to confirm none reports an oracle or revisit-stratified results.

GO requires all three of the following:
(a) On at least 5 of 8 clips, the best heuristic is >=1.5x worse than the full cache on long-gap revisit frames in AbsRel or RPE, while staying within 10% on novel frames.
(b) The oracle recovers >=50% of that loss.
(c) Cumulative-attention AUROC is <=0.65 on later-long-gap-revisited tokens and >=0.8 on other tokens.

**Kill criterion.**

Drop the project if any of the following holds:
- At both 25% and 10% budgets, the best existing heuristic stays within 1.2x of the full cache on long-gap revisit frames on most clips, so there is no revisit-specific failure.
- The closed-loop oracle beats the best heuristic by <15% on every stratum, so there is no headroom to audit.
- Attention-history AUROC on later-revisited tokens is >=0.85, so persistence of importance holds.
- The full STAC or RetrieveVGGT paper already reports an oracle comparison together with revisit-stratified evaluation.

If the full-cache model itself collapses on long-gap revisits, which would make eviction policies indistinguishable, pivot the paper to a cross-store comparison (KV cache vs TTT3R state) instead of continuing the eviction audit.

**Compute.** 140 GPU-hours over 12 weeks. One 80 GB A100/H100. The full-cache KV costs about 100 MB per frame, so 500 frames is about 50 GB plus a 5 GB model. Bounded-budget runs and <=200-frame full-cache clips also fit on a 24-48 GB card. A CPU workstation handles revisit labelling, AUROC analysis and the water-filling analysis, plus about 1-2 TB of disk for sparsified attention logs (about 6 GB per 500-frame clip before sparsification).

**Risks.**

- Soft-attention oracle is not a strict optimum. Evicting tokens changes later attention, so the open-loop oracle may be unachievable. Mitigation: a closed-loop iteration, and calling it an 'empirical oracle reference' rather than a bound.
- Attention mass is not causal importance. Mitigation: a value-norm-weighted variant, plus group-knockout ablations on 5 clips to check that oracle utility tracks the output-error increase.
- Full-cache StreamVGGT may itself degrade on long sequences. Bounded caches can beat it, as several prior works report, so the full cache is a reference, not an upper bound. Report the oracle against both.
- Long-gap revisits are rare in standard benchmarks, so loop clips must be curated and the A-B-A streams are partly synthetic, which invites selection-bias critique. Release the labels and the selection rule.
- Place cells built from drifted predicted points can mis-bin tokens. Ablate voxel size, direction bins and q_min, and compare against GT-point cells as an upper bound.
- Most competitor methods may lack code. The '-lite' reimplementations of their scoring rules must be faithful and disclosed, and RetrieveVGGT and STAC must be run without CPU offload to keep budgets equal, which may be argued as unfair to them.
- Crowded field moving monthly (RegVGGT and IncVGGT appeared within weeks of each other). A competitor may publish an oracle or stratified evaluation first, so the week-1 PDF check and a fast arXiv release matter.
- The PQ fix may give only small gains over STAC or RetrieveVGGT. The paper must still stand on the audit, the benchmark and the analysis.

**Target venue.** ICCV 2027 (deadline about March 2027, which fits the 12-week plan). Fallback: NeurIPS 2027 Datasets and Benchmarks track, given the audit and benchmark framing. A 4-page version could go to a CVPR 2027 workshop on continual or embodied 3D perception.

**First-round prior-work verdict: partially_novel.** Place-aware eviction alone is not defensible. RetrieveVGGT already thins over-populated camera-pose grid cells to handle revisits, STAC already keeps a per-voxel capped long-term cache keyed on predicted 3D points, and OVGGT already protects view-overlap anchors. The paper is therefore repositioned as an audit and benchmark with PQ as the minimal CL-derived intervention. What remains unclaimed:
(i) A closed-loop future-attention oracle for streaming 3D transformers, with a 'fraction of oracle gap closed' score for every published policy. Such oracles exist only for LLMs: ForesightKV, AgentKV and LU-KV.
(ii) A direct falsification test of persistence of importance under revisits, targeting RegVGGT's stated assumption.
(iii) Revisit-gap stratification by GT frustum overlap. 2608.27529 only reports multi-gap RPE on KITTI.
(iv) A water-filling account of why balanced buffers should help.

Caveats: PDFs could not be opened in the novelty check, and this step's web-search budget was exhausted. Only the code and checkpoints of StreamVGGT, STream3R, Point3R, CUT3R, TTT3R and VGGT were verified, via git. The week-1 PDF read is therefore mandatory.

**Closest work found in the first round:**

- [Attention Itself Could Retrieve. RetrieveVGGT: Training-Free Long Context Streaming 3D Reconstruction via Query-Key Similarity Retrieval (arXiv 2605.09644)](https://arxiv.org/pdf/2605.09644) — arXiv 2026 (May). Most threatening for the fix. According to search snippets, it keeps a pose-aware spatial memory. Each frame's estimated camera position and optical-axis direction place it in a K x K x K grid over the bounding box of camera positions seen so far. The method is explicitly motivated by revisits that fill the cache with near-duplicate keyframes. It periodically thins over-populated cells with evenly spaced temporal subsampling and leaves under-represented cells intact. That is essentially 'evict from the fullest place cell, protect rare places'. Memory is bounded through a fixed frame budget (48), tombstones and a 0.5 deletion ratio. Differences: the cells are camera-pose cells at frame granularity, not token-level cells built from predicted 3D points crossed with viewing direction. It has no explicit minimum quota. It is pitched as retrieval rather than as a CL balanced buffer. I could not confirm whether it offloads KV to CPU.
- [STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction (arXiv 2603.20284)](https://arxiv.org/html/2603.20284) — CVPR 2026 (Highlight). Bins evicted tokens into a 3D voxel grid by their predicted 3D position. Each voxel keeps a staging buffer plus a capped set of long-term representatives, with merge and re-merge when full. This is a per-voxel capacity on the cache, and the snippets say it explicitly lets the model retrieve earlier evidence when the camera returns to a region. The working cache uses decayed cumulative attention, a first-frame anchor and a sliding window. It overlaps strongly with 'tag tokens by a voxel of the predicted point and cap per cell', and it is plug-and-play on causal VGGT. Differences: it merges instead of using hard quota eviction, it has no viewing-direction bins, and I saw no oracle bound or revisit-stratified evaluation.
- [OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer (arXiv 2603.05959)](https://arxiv.org/pdf/2603.05959) — arXiv 2026 (Mar, v3 Apr). Uses a fixed-budget KV cache and protects first-frame tokens as a global anchor. It also registers historical anchors adaptively, based on view-overlap coverage, to supply long-range geometric references. This is a co-visibility-driven protection of old tokens, which overlaps with the idea's co-visibility predictor and frame-0 protection. Its scoring uses FFN residual magnitudes. No oracle or revisit stratification was found.
- [GHOST: Geometry-Hierarchical Online Streaming Token Eviction for Efficient 3D Reconstruction (arXiv 2605.15852)](https://arxiv.org/pdf/2605.15852) — arXiv 2026 (May). A training-free eviction method that uses the model's own geometry outputs: camera pose change, depth-gradient variance, depth confidence and recency. It protects special tokens and argues that query-dependent attention signals track the long-term geometric value of 3D tokens poorly. That partly pre-empts the 'persistence of importance fails' critique. It has no place quotas and no oracle.
- [Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction (arXiv 2605.05749)](https://arxiv.org/pdf/2605.05749) — arXiv 2026 (May). Covers the Point3R second case. It adds viewing direction and timestamps to pointer memory and uses spatial distance plus viewing angle to tell redundant observations from revisits. It compares retain, replace, random and merge policies, and runs pose refinement on loop detection. This overlaps with place cells defined as voxel x viewing direction for pointer memory.
- [ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution (arXiv 2602.03203)](https://arxiv.org/pdf/2602.03203) — ICML 2026 (PMLR v306). 'Golden Eviction' builds an oracle from future attention scores, evicts the KV pairs with the lowest future score and states an upper-bound argument. This is the LLM version of the Belady-style future-attention oracle. AgentKV (2609.14872) and LU-KV (2602.08585) likewise define offline future-utility oracles and measure the recall of heuristics against them. So the audit methodology exists, but only for LLMs.
- [Evict3R / InfiniteVGGT / StreamCacheVGGT / FrameVGGT / RegVGGT (2509.17650, 2601.02281, 2604.15237, 2603.07690, 2609.23286)](https://arxiv.org/pdf/2609.23286) — 2025-2026. Importance-, redundancy- or saliency-based bounded caches for StreamVGGT. RegVGGT (Sept 2026, new) explicitly assumes that a token's initial saliency dictates its long-term importance, which is exactly the persistence assumption the idea wants to test. It evaluates on ScanNet with 800-1000 frames against Evict3R and InfiniteVGGT. These are baselines, not scoops, but the field is crowded and now includes RegVGGT and IncVGGT (ICLR 2026).
- [Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction (arXiv 2608.27529)](https://arxiv.org/html/2608.27529) — arXiv 2026 (Aug). Reports multi-gap relative pose error as the temporal gap grows on KITTI, a partial form of gap-stratified evaluation. It argues that local context (11 frames) plus a loop-closure backend suffices. This weakens the claim that nobody stratifies by gap, though it does not stratify by GT frustum overlap or test eviction policies.

---

## 13. What Do Streaming 3D Models Remember? Continual-Learning Accuracy Matrices, Scene-Change Plasticity Tests, and a Delta-Rule Reset Law for Streaming 3D Memory

**Judge:** overall 4.5/10, tier C. Scores out of 5: novelty 2, low compute 4, impact 3, speed of first signal 4.
Its read-only probes would be useful instruments, but forgetting in the CUT3R state, the roughly 3-frame horizon, retention theory and revisit write suppression are all already claimed (TTSA3R, MeMix, ReCal3R, FILT3R, 2605.16981, HorizonStream). Fold its probes into P1 and P4 as diagnostics rather than running it as a paper.

**Biggest weakness:** Raymap and re-present probes may measure priors rather than memory. The delta-rule reset-law prediction (Spearman 0.6 over 20 or more sequences) is fragile, and the benchmark framing has little left that is new.

**One line.** A forward-pass-only benchmark that probes streaming 3D memories read-only, filling continual-learning accuracy matrices for each memory type. It adds scene-switch and 3RScan rescan plasticity tests, and fits a delta-rule model to CUT3R tokens to predict the best reset period and a write-protect-on-revisit gate.

**Memory store.** Fast state: the recurrent state tokens of CUT3R and TTT3R, as the main subject. Contrasted with KV-cache memory (StreamVGGT, STream3R in causal mode and with eviction budgets), explicit spatial and pointer memory (Spann3R, Point3R), and a local window with no persistent memory (STream3R window mode). Three management decisions are studied: when to reset, whether to write on a revisit, and how large a KV budget to keep.

**Task.** Streaming 3D reconstruction of indoor RGB video: online pointmaps and camera poses. The paper measures and manages what the streaming memory retains, forgets and updates. Data:
- 7-Scenes and NRGBD under TTT3R's 300/500/1000-frame protocol.
- TUM RGB-D loop sequences.
- About 30 ScanNet v2 scenes.
- About 30 A-B-A streams stitched from non-overlapping 7-Scenes and ScanNet scenes.
- About 50 3RScan reference-to-rescan streams with annotated rigid object moves and removals.

**Gap in the big picture.** The project tests three lessons from the big-picture document in streaming 3D, and adds the document's open consolidation problem:
- Forgetting tracks how far an update moves the model (sparse memory finetuning 2510.15103; RL's Razor). This becomes a displacement-versus-elapsed-frames law for state memory.
- Continuously updated memories can end up worse than no memory (2605.12978). This becomes a measured crossover beyond which memory-on probes of early frames lose to a no-memory prediction.
- Old and new compete for finite memory (2604.27003). This becomes A-B-A interference.
- The open consolidation question of when to keep or discard state becomes a theory-predicted reset period and a revisit write rule.

Streaming-3D papers report only end-of-sequence ATE and Acc/Comp. 2605.16981 infers a ~3-frame horizon from gate magnitudes, not from stored content. HorizonStream gives an exponential-decay kernel to motivate a new architecture. ABot-Recon argues from ATE that persistent memory loses to a local window. Nobody measures, per past view, what is retained, when it is lost, or whether memory updates when the world actually changes. TTT3R's reset period (N=100) is a heuristic.

**Hypothesis.** On 7-Scenes, NRGBD, TUM and ScanNet streams, two things will hold. First, cumulative state displacement since frame i will explain the probed forgetting of frame i in CUT3R-family states better than elapsed frames do (R^2 gap >= 0.2). Second, a delta-rule retention model fitted to CUT3R tokens will predict each sequence's empirically best TTT3R reset period (Spearman >= 0.6 over >= 20 sequences).

**Method.**

(1) Probes. Every D=10 frames, snapshot each model's memory: CUT3R/TTT3R state tokens, the StreamVGGT/STream3R KV cache, or the Spann3R/Point3R spatial memory. Run two read-only probes on a fixed set of 20 past frames i<t.
- P1, re-localisation (all architectures): re-present image i with no write (CUT3R update=False, no cache append, no memory write). Score world-frame pointmap Chamfer and pose error after one Sim(3) fitted to the online trajectory up to t.
- P2, content recall (CUT3R family only): query a raymap at frame i's GT-aligned pose with no image. Score depth AbsRel and delta<1.25.
- Control: a matched query issued before region i was ever observed, which subtracts learned priors.

(2) Accuracy matrices. The probes fill A[t,i], from which we report forgetting, BWT, retention half-life and AUC. We also report memory benefit against a no-memory pair prediction and against STream3R window=5 (an ABot-Recon-style local context).

(3) Plasticity tests. A-B-A streams measure retention of A after B, pollution of B and frames to re-acquire A. 3RScan reference-to-rescan streams, primed with the reference scan, measure:
- ghost rate: the fraction of moved or removed object surface still predicted at its old location;
- update latency in frames;
- corruption of unchanged regions, as the control.

(4) Displacement law. Regress per-frame forgetting on cumulative state displacement since i (token L2 and CKA for states, KV turnover, pointer churn) and, separately, on elapsed frames. Compare the probed half-life directly with the gate-derived horizon of 2605.16981, computed from the same runs.

(5) Theory. Model TTT3R's update as delta-rule regression, S <- S - beta(Sk - v)k^T, over state tokens. It yields three predictions:
- Retention after tau unobserved frames is roughly exp(-beta*mu*rho*tau), where mu is key coherence and rho is the novelty rate.
- The reset period N* that minimises cumulative error trades drift accumulated in the state (proportional to sigma_d*N) against the re-warm cost of each reset (proportional to c/N). This gives N* ≈ sqrt(2c/sigma_d), capped by a multiple of the half-life.
- A drift-overwrite condition: on a revisit (high key alignment), set the per-token learning rate to 0 when the incoming frame's estimated drift exceeds the stored token's.

(6) Fit and test. Fit beta, mu, rho, sigma_d and c from CUT3R tokens on held-out sequences. Test the predicted N* against an empirical grid {25, 50, 100, 200, 400, inf}. Run the write-protect gate, which costs nothing extra at inference, head-to-head against TTSA3R's spatial gate, ReCal3R and the 2605.16981 frame gate.

**Datasets.**

- 7-Scenes (Microsoft RGB-D 7-Scenes), TTT3R 300/500/1000-frame protocol
- Neural RGB-D (NRGBD) scenes, TTT3R protocol
- TUM RGB-D: loop sequences such as fr1_room, fr2_desk, fr3_long_office_household, plus exploration sequences as the no-revisit control
- ScanNet v2: about 30 validation/test scenes, preprocessed with the Spann3R/MonST3R scripts
- A-B-A switch streams stitched from non-overlapping 7-Scenes and ScanNet scenes (about 30, built by us)
- 3RScan (github.com/WaldJohannaU/3RScan): about 50 reference/rescan pairs with RGB-D sequence.zip, camera poses, rescan-to-reference transforms and annotated rigid/removed instances; released as a streaming-change split

**Models / checkpoints.**

- CUT3R: cut3r_512_dpt_4_64.pth (github.com/CUT3R/CUT3R, verified); 224 linear checkpoint optional
- TTT3R: same CUT3R checkpoint with --model_update_type ttt3r and --reset_interval (github.com/Inception3D/TTT3R, verified)
- TTSA3R, MeMix, ReCal3R and the 2605.16981 parameter-free gates: training-free update rules on the same CUT3R checkpoint (official code where released, otherwise reimplemented from the papers)
- StreamVGGT: Hugging Face lch01/StreamVGGT (github.com/wzzheng/StreamVGGT, verified)
- STream3R: Hugging Face yslan/STream3R, in causal (KV cache) and window (size 5, local context) modes (github.com/NIRVANALAN/STream3R, verified)
- Spann3R: spann3r.pth (github.com/HengyiWang/spann3r)
- Point3R: released Google Drive checkpoint (github.com/YkiWu/Point3R, verified)
- InfiniteVGGT / Evict3R KV-eviction budgets on StreamVGGT, if code is available
- Offline references: facebook/VGGT-1B (upper bound on 300-500-frame clips) and optionally yyfz233/Pi3

**Baselines.**

- Per-chunk reset / no-memory pair prediction (lower bound on memory benefit)
- STream3R window=5 (local context without persistent memory, standing in for ABot-Recon 2608.27529)
- CUT3R vanilla state update
- TTT3R with the heuristic reset N=100, and without reset
- 2605.16981 frame-level parameter-free gate (gate-derived horizon)
- TTSA3R spatial-coverage gate (closest to write-protect-on-revisit)
- ReCal3R reliability-calibrated learning rate
- MeMix
- FILT3R Kalman gain (if code is released)
- StreamVGGT and STream3R full KV cache, plus eviction budgets
- Spann3R and Point3R explicit memory
- VGGT-1B offline (upper bound)

**Metrics.**

- Accuracy matrix A[t,i] for probe P1 (world-frame and Sim(3)-aligned Chamfer; RRA/RTA pose error) and probe P2 (depth AbsRel, delta<1.25)
- Forgetting F_i = max_t A[t,i] - A[T,i]; backward transfer (BWT); retention half-life (exponential fit) and retention AUC
- Memory benefit = error without memory minus error with memory, net of the before-observation prior control; crossover length where memory benefit < 0
- A-B-A: retention of A after B, interference on B, frames to re-acquire A
- 3RScan: ghost rate on moved/removed instances, update latency (frames until ghost rate < 20%), unchanged-region corruption
- Displacement law: R^2 of forgetting on cumulative displacement vs on elapsed frames; ratio of probed half-life to gate-derived horizon
- Spearman correlation between predicted and empirically best reset period
- Standard metrics: ATE/RPE, Acc/Comp/NC; Kendall tau between the ATE ranking and the retention ranking of methods

**First experiment (go/no-go).**

Weeks 1-2 on one A100-80GB, about 8 GPU-hours.

Setup: install CUT3R and TTT3R with cut3r_512_dpt_4_64.pth. Run 3 7-Scenes 1000-frame sequences (TTT3R protocol) and TUM fr3_long_office_household (a loop sequence) under four configurations: CUT3R, TTT3R without reset, TTT3R with reset 100, and per-chunk reset. Snapshot the state every 10 frames, offloading it to CPU.

Probes: run P2 (raymap query at frame i's GT-aligned pose, update=False) and P1 (re-present frame i, update=False) on 20 fixed past frames. Add the before-observation prior control and a no-memory pair prediction. From the same TTT3R runs, compute the gate-derived horizon of 2605.16981.

Week 2 analysis:
- Fit per-config half-lives with bootstrap CIs across sequences.
- Check probe sanity: at lag 0, the P1 prediction must match the online prediction within 5% AbsRel, and the P1 result must be stable when frame i is swapped for i±1.
- Run a first displacement-versus-time regression.

Go if all three hold:
- At lags <= 10, P2 beats the prior control by >= 10% relative AbsRel, so the probe measures memory rather than priors.
- The half-lives of CUT3R, TTT3R and per-chunk reset separate beyond their 95% CIs.
- The probed half-life is either reported directly or differs from the gate-derived horizon by a measurable factor.

**Kill criterion.**

Drop the project if, in the pilot, two things are both true:
- The CUT3R raymap probe beats the before-observation prior control by less than 10% relative AbsRel at lags <= 10 frames.
- The half-lives of CUT3R, TTT3R and per-chunk reset cannot be separated beyond across-sequence 95% CIs.
Together these mean the probes have no discriminative power.

Drop the theory-and-policy half and publish the benchmark alone if either holds at week 11:
- The fitted delta-rule model predicts the empirically best reset period with Spearman < 0.3 across >= 20 sequences.
- Write-protect-on-revisit fails to match TTSA3R/ReCal3R on TUM and 7-Scenes loop ATE while also leaving exploration sequences unchanged.

**Compute.** 140 GPU-hours over 14 weeks. 1x A100/H100 80 GB, all forward passes with no training. A 48 GB card (L40S/A6000) works if KV-cache models are capped at 300-frame clips. The theory, simulations, regressions and model fitting run on CPU with NumPy.

**Risks.**

(1) Probe semantics differ across architectures. Only the CUT3R family supports raymap recall (P2), so cross-architecture claims rest on the re-localisation probe P1 with a single shared Sim(3) protocol. CUT3R's update=False and ray_mask flags need confirming in its inference code in week 1.

(2) At long lags, global pose drift can dominate probe error. Report world-frame and Sim(3)-aligned versions and use the prior control.

(3) KV models run out of memory on long streams. Cap them at 300-500-frame clips or use eviction budgets, and state this caveat explicitly.

(4) 3RScan data issues:
- access requires a terms-of-use form;
- the Tango imagery is low-resolution;
- rescans differ in trajectory and lighting.
Unchanged regions serve as the control.

(5) The linear delta-rule model may capture only orderings and scalings for a nonlinear transformer state. Validate it on predicted rank orders, not absolute values.

(6) TTSA3R, ReCal3R, ABot-Recon and FILT3R may have no released code. Reimplement the training-free rules and use STream3R window mode as the local-context reference.

(7) Scoop risk is high because the area moves monthly. Put the benchmark and probe toolkit out as an arXiv preprint around week 8.

(8) Related-work overlaps come from snippets only, since arXiv could not be opened. Re-verify them against the PDFs before writing.

**Target venue.** ICCV 2027 main track, as an analysis-and-benchmark paper (deadline around March 2027), with an arXiv preprint of the benchmark around January 2027. Fallback: NeurIPS 2027 Datasets & Benchmarks or 3DV 2027.

**First-round prior-work verdict: partially_novel.** Already claimed: the CUT3R state forgets (TTSA3R, MeMix, ReCal3R, FILT3R); a ~3-frame gate-derived horizon (2605.16981); exponential-retention theory serving a new architecture (HorizonStream); local window beating persistent memory by ATE (ABot-Recon); revisit write suppression (TTSA3R, ReCal3R).

What remains defensible:
- read-only content probes that fill continual-learning accuracy matrices across recurrent, KV and pointer memories;
- A-B-A and 3RScan rescan plasticity tests on feed-forward streaming memory;
- the displacement-versus-time law;
- a delta-rule model fitted to CUT3R tokens that predicts the reset period, with write-protect claimed only as a theory-derived member of the existing gating family.

Caveat: both this pass and the novelty check could not open arXiv (DNS and proxy failures), and the search budget ran out. The related-work overlaps above come from snippets and must be re-checked against the full PDFs.

**Closest work found in the first round:**

- [Rethinking the State Update Gate for Long-Sequence Recurrent 3D Reconstruction (Ren et al.)](https://arxiv.org/pdf/2605.16981) — arXiv 2605.16981, May 2026. The closest diagnostic paper. It measures TTT3R-style gate statistics across five benchmarks (median 0.31, never above 0.6, nearly frame-invariant), converts them into an effective memory horizon of about 3 frames per state token, and blames this for long-sequence drift. It then proposes parameter-free frame-level gates (51% lower ATE on TUM). So 'the recurrent state has a short horizon and that explains drift' is already claimed. Its horizon comes from gate magnitudes, not from read-only content probes, and it has no cross-architecture accuracy matrix, no scene-change test and no reset theory. This is from abstract and snippet text only; the full PDF could not be opened.
- [TTT3R: 3D Reconstruction as Test-Time Training](https://arxiv.org/pdf/2509.26645) — ICLR 2026 (arXiv 2509.26645). Frames the CUT3R state update as test-time online learning with a closed-form, confidence-derived learning rate, which is the online-regression view. It also reports an empirically chosen reset period (N=100), used only for sequences over 1000 frames. It gives no retention law, no derivation of the optimal reset period and no delta-rule capacity analysis. The idea's theory extends this view directly.
- [HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction](https://arxiv.org/pdf/2605.23889) — arXiv 2605.23889, May 2026. Includes a retention-style theory for streaming 3D (an 'evidence influence kernel'). With no forgetting, the initial state keeps full magnitude, which the authors call the root cause of degradation in TTT without reset, in CUT3R and in linear attention. With per-channel retention below 1, its contribution decays exponentially. This partly overlaps the idea's delta-rule account of exponential retention and reset. However, it is a new-architecture paper (geometric linear attention), not a probe benchmark. Seen only in search snippets.
- [Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction (ABot-Recon)](https://arxiv.org/pdf/2608.27529) — arXiv 2608.27529, Aug 2026. Argues that persistent long-range learned memory is not needed for long streams. A 12-frame local KV window plus chained relative poses gives about 40% less drift than persistent-memory methods on Oxford Spires/KITTI/VBR. The authors note that persistent context helps more in confined indoor scenes with revisits. This partly anticipates hypothesis (3), that memory can become worse than no memory past a crossover length. It does so through end-of-sequence ATE, not content probes.
- [ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction](https://arxiv.org/pdf/2607.05356) — arXiv 2607.05356, Jul 2026. Another training-free per-token learning-rate rule for CUT3R. It uses state reconstruction residual and recent 'update pressure' to stop reliable history from being overwritten (3.7x lower ATE). This overlaps the write-protect policy direction (suppress writes that would corrupt reliable state). It offers no retention measurement or theory.
- [FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction](https://arxiv.org/pdf/2603.18493) — arXiv 2603.18493, Mar 2026. Uses a Kalman gain with per-token variance, so process noise rises with genuine scene change. It explicitly frames the stability-plasticity tradeoff, and it critiques resets for discarding context and causing scale discontinuities. This is principled update theory, but there is no forgetting curve, no derivation of the reset period and no rescan or scene-change benchmark.
- [MeMix / TTSA3R / Mem3R / PAS3R (CUT3R-state update fixes)](https://arxiv.org/pdf/2603.15330) — arXiv 2603.15330 / 2601.22615 / 2604.07279 / 2603.21436, 2026. Each motivates its method with 'catastrophic or temporal forgetting' of the fixed-size state, and each reports end-of-sequence metrics over 300-1000 frame streams on 7-Scenes, NRGBD and TUM. TTSA3R uses a spatial module that checks whether a region was already observed before allowing updates, which is close to write-protect-on-revisit. None of them probes stored content per past frame.
- [InfiniteVGGT (Long3D benchmark) / FrameVGGT / Evict3R / XStreamVGGT](https://www.alphaxiv.org/abs/2601.02281.md) — arXiv 2601.02281 / 2603.07690 / 2602.21780, 2026. Long3D is a 2k-10k-frame long-stream benchmark scored only with global Acc/Comp/CD/NC. The KV-eviction budgets supply the benchmarked KV-memory variants. None of these papers measures revisit consistency, per-frame retention or scene-change plasticity.
- [Objects Can Move / Has Anything Changed? / OASIS-Map (3RScan change detection)](https://arxiv.org/pdf/2607.14899) — 2022 / arXiv 2312.01148 / arXiv 2607.14899 (2026). 3RScan reference/rescan change detection exists, but only for offline or TSDF multi-session mapping pipelines. None applies it to feed-forward streaming memory (CUT3R/StreamVGGT/Point3R). This supports the novelty of the plasticity test.

---

## 14. Beyond Fact Admission: Routing Facts, Skills, Format Shifts and Noise Across Context, Store and LoRA, with Corroboration-Gated Weight Writes

*Added in the second round for an angle the first round missed.*

**Judge:** overall 4.5/10, tier C. Scores out of 5: novelty 2, low compute 3, impact 3, speed of first signal 4.
It speaks to Harrington's 'type of change decides the store' directly. But Dual-Layer Agentic Memory owns fact routing plus write-back, and the destination-by-type matrix risks confirming the obvious (skills need weights, formats fit in a prompt). The corroboration gate is the most novel piece and could become a section of P9 or P5.

**Biggest weakness:** The kill criterion (b), where the full router lands within 1 point of the s2-only Dual-Layer analogue plus always-LoRA, is likely to trigger. The mixed stream is artificial, the re-implementation of an unreleased baseline invites 'unfaithful' critiques, and there is high scoop pressure from the Dual-Layer authors.

**One line.** Dual-Layer Agentic Memory (Aug 2026) already routes fact writes to an external store and consolidates them into weights. This project extends write-time routing to updates that are not facts (new skills, output-format shifts, and noisy or poisoned packets), adds a context-rule destination and a quarantine-then-corroborate gate before any weight write, and tests at 0.5-1.5B scale whether a destination-by-type matrix and cheap forward-pass signals beat the fact-only router and always-LoRA.

**Memory store.** Cross-store write policy. Each update gets one of five actions: quarantine, a context rule slot (pinned, capped at about 512 tokens, oldest evicted), the external store (BM25 with timestamped overwrite), a LoRA r=8 adapter (with replay from a small data buffer), or the store plus LoRA. A weight write is allowed only after the corroboration gate passes. Store-to-weight consolidation is no longer a claimed contribution, because Dual-Layer Agentic Memory and Dennis et al. do it. Here it is a re-implemented baseline and ablation. The project targets the big_picture.md open problem of consolidation across stores, framed as which store a given kind of change belongs in.

**Task.** Sequential adaptation of Qwen2.5-0.5B/1.5B-Instruct to one mixed stream with five change types. (1) New facts: EvolvingQA new split and WikiBigEdit timesteps. (2) Revised facts: EvolvingQA updated split. (3) New skills: TRACE C-STANCE, NumGLUE-cm and Py150, each as a packet of 64-500 demonstrations. (4) Format shifts: the output schema of an already-solved ScienceQA or FOMC task changes. (5) Noise: plausible same-type entity swaps of real updates at rates of 0, 10 and 30%, plus a shuffled-label FOMC chunk. Probes run after every stage. Retention is measured on EvolvingQA's invariant split and on earlier TRACE tasks.

**Gap in the big picture.** Write-time routing between stores already exists for facts. Dual-Layer Agentic Memory (Li et al., arXiv 2608.22215, Aug 2026) has a fast write router with three labels: non-write if the model already answers correctly, write-new if it refuses zero-shot but answers given the fact, and write-update if it is confidently wrong. Its 1.7B/8B cascade prunes up to 68% of redundant external writes, and slow SFT write-back moves high-value entries into the weights. D-MEM (2603.14597) gates memory evolution on surprise or reward-prediction error. Nightly LoRA consolidation (Dennis et al., 2605.24657) retains 80% of knowledge vs 37% for context compaction. Harrington et al. (2607.07847) show that the type of environmental change decides whether adaptation needs the weights, but they run each method on the whole stream. The open part is the following. (a) The routers above handle facts or conversational turns and target only the external store, then consolidate later. None decides at write time among context rule, store and adapter for updates that are not facts: skills that need weights and format conventions that a prompt rule may carry. (b) Weight-consolidation pipelines have no poisoning or corroboration gate; the write gates in MAPLE-Guard and the budget-curated memory work protect external stores only. (c) Nobody reports a destination-by-type matrix measured under one protocol at small scale, with cost. The revised project fills (a)-(c) and uses the fact-only router as its main baseline.

**Hypothesis.** H1 (the destination-by-type matrix is not degenerate, and the part that matters is not about facts): at 1.5B, the best destination differs by type by at least 5 points. Specifically: format shifts reach at least 90% of LoRA's gain with a context rule and zero gradient steps; skills need LoRA (at least 10 points over store or context); noise is best dropped. H2 (more than fact-only routing): with no type labels, a router on s1-s6 beats a re-implemented Dual-Layer fact router (no/new/update writes to the store plus periodic SFT write-back) by at least 3 points on the stream-average score. Most of the gain must come from skill and format packets. The router must also use at most 35% of always-LoRA's gradient steps and reach functional-type macro-F1 of at least 0.7. H3 (gated weight writes): at 30% noise, quarantine plus corroboration before LoRA cuts retention loss and poisoned-answer rate by at least 50% vs always-LoRA and vs ungated Dual-Layer-style write-back, with at most 2 points lost on clean updates.

**Method.**

Packets: a fact passage or QA pair, K demonstrations, or a rule statement. Signals come from the frozen model plus the store, at 2-4 forward passes per packet. (s1) Surprisal: closed-book target NLL; this also stands in for the D-MEM surprise score. (s2) In-context gain: the change in target log-prob on a self-generated paraphrase probe with vs without the packet in context. Its fact-only special case is the Dual-Layer new/update test, so s2 alone, restricted to facts, reproduces that router. (s3) Store conflict, from NLI or an LLM judge, plus parametric stubbornness: the model gives the old answer even with the new fact in context. (s4) Corroboration: the count of independent sources asserting the claim within a window. (s5) Template regularity: multiple input-output pairs sharing one schema. (s6) Format-only error: correct after normalization but failing exact schema match. Routers: (a) a training-free decision tree; (b) logistic regression fit on one disjoint calibration stage; (c) an API zero-shot classifier. Actions: quarantine; context rule; store append or overwrite; LoRA (per stage, trained on routed packets plus buffer replay). The gate: any packet bound for LoRA waits in quarantine until s4 reaches c (with c=1 as an ablation) or its store entry is retrieved and goes uncontradicted m times. Analyses: the destination-by-type matrix at 0.5B, 1.5B and the API model (where the boundary sits); signal ablations (s2-only, which is the Dual-Layer analogue, vs the full set); a noise-rate sweep; and a cost accounting of gradient steps, context tokens and store size. An optional follow-up outside the budget: the same novelty, conflict and corroboration gate between CUT3R's recurrent state and Point3R's explicit memory, in the spirit of TTT3R's alignment-based learning rate.

**Datasets.**

- EvolvingQA (Carpe Diem, NAACL 2024): invariant, new and updated splits; https://arxiv.org/abs/2311.08106
- WikiBigEdit (ICML 2025), subsampled to about 1-2k edits per timestep; https://arxiv.org/abs/2503.05683
- TRACE (arXiv 2310.06762): C-STANCE, FOMC, Py150, ScienceQA, NumGLUE-cm and NumGLUE-ds, plus the LIMA replay set
- Harrington et al. scenarios (factual updates, temporal drift), reused if the code at github.com/anneharrington/studying-cl exposes them; https://arxiv.org/pdf/2607.07847
- Built by this project with fixed seeds: format-shift packets on ScienceQA and FOMC, and noise and poison packets (entity swaps, shuffled labels) at 0, 10 and 30%

**Models / checkpoints.**

- Qwen/Qwen2.5-1.5B-Instruct: primary model for LoRA r=8 and the signals
- Qwen/Qwen2.5-0.5B-Instruct: scale check
- One pinned, low-cost API chat model: the context-and-store-only arm, the zero-shot router, and the conflict judge
- BM25 retriever; a small sentence-embedding model as an ablation

**Baselines.**

- Do nothing (frozen base model)
- Always-context: an ACE/GEPA-style bounded playbook
- Always-store: append-only RAG, and RAG with overwrite
- Always-LoRA with and without LIMA replay; GainLoRA (https://arxiv.org/pdf/2505.15424)
- Re-implemented Dual-Layer Agentic Memory fact router (non-write/write-new/write-update to the store) plus periodic SFT write-back (https://arxiv.org/pdf/2608.22215): the key baseline
- D-MEM-style surprise-gated writes (https://arxiv.org/pdf/2603.14597)
- Nightly LoRA consolidation of the whole buffer (Dennis et al., https://arxiv.org/pdf/2605.24657)
- UltraEdit or MEMIT for fact packets
- Store + LoRA on every packet (union policy)
- Random router with matched action proportions
- Oracle-type router and oracle-destination router (upper bounds)

**Metrics.**

- EvolvingQA EM/F1 per split, plus the outdated-answer rate
- WikiBigEdit efficacy, rephrase generalization and locality
- TRACE average performance and backward transfer with native scorers
- Format-compliance rate after format shifts
- Poisoned-answer rate, false-write rate, and retention drop vs noise rate
- Stream-average score, and the gain over the Dual-Layer fact router split by change type
- Router macro-F1 on functional type, and regret vs the oracle-destination router
- Cost: gradient steps and GPU-seconds per 1k packets, store size, context tokens per query, API dollars

**First experiment (go/no-go).**

Ten days on one GPU with Qwen2.5-1.5B-Instruct. Days 1-2: build a mini stream of 300 EvolvingQA new, 300 updated and 300 invariant facts; 3 TRACE skills with 500 train and 200 test examples each; 2 format shifts; 20% noise. Re-implement the Dual-Layer three-way fact router from its published label definitions. Days 3-6: fill the destination-by-type matrix with 5 actions x 5 types x 3 seeds. Days 7-8: compute s1-s6 and fit the tree and logistic routers on a disjoint calibration split. Days 9-10: run the full mixed stream in 2 orders x 3 seeds. Compare the oracle-type router, the signal routers, the Dual-Layer fact router plus write-back, always-store and always-LoRA, with the corroboration gate on and off. GO if all hold: (i) the oracle-type router beats the best single destination by at least 3 points (95% bootstrap CI excluding 0); (ii) the signal router beats the Dual-Layer fact router by at least 2 points, with the gain concentrated on skill and format packets; (iii) the gate cuts the poisoned-answer rate by at least 30% at 20% noise. About 20 GPU-hours and $50 of API calls.

**Kill criterion.**

Kill the project if any of these holds. (a) Context rules fail on format shifts at both 0.5B and 1.5B (under 60% of LoRA's gain), and skills and facts share a best destination, so the matrix collapses to the store for facts and LoRA otherwise. (b) The full-signal router is within 1 point of the s2-only fact router (the Dual-Layer analogue) plus always-LoRA for packets that are not facts. Then nothing beyond Dual-Layer survives. (c) The corroboration gate does not reduce the poisoned-answer rate, or it costs more than 3 points on clean updates. If (a) holds only at 1.5B but the API arm shows a clear context-vs-store split, write a short note on scale dependence instead.

**Compute.** 130 GPU-hours over 8 weeks.

**Risks.**

(1) Scoop pressure. Dual-Layer Agentic Memory covers fact write routing and consolidation, and its code is not yet released. The re-implementation from the paper's label definitions may be judged unfaithful; mitigate by releasing the re-implementation and matching their label rules exactly. A follow-up by those authors could add skills, formats or noise in the coming months, so move fast. (2) Harrington et al. may already contain per-scenario recommendations close to H1. Read their full paper and code first, and position H1 as a small-scale, per-packet measurement with cost. (3) At 1.5B, in-context learning may be too weak for context rules to carry format shifts, which would make the matrix degenerate (the API arm mitigates this). (4) Noise packets may be trivially or impossibly detectable; calibrate with plausible swaps. (5) Corroboration delays legitimate single-source updates, so report the latency-accuracy trade-off. (6) The mixed stream is artificial; release the generators. (7) Judge bias in the conflict signal; ablate with NLI-only conflict scoring. (8) The 3D extension stays speculative and outside the budget.

**Prior-work verdict: partially_novel; recommendation: pursue_with_pivot.** The original plan's fact routing (new vs revised vs none, from zero-shot vs in-context behavior) and its Phase 2 store-to-weights consolidation are scooped by Dual-Layer Agentic Memory (Aug 2026). Nightly LoRA consolidation (2605.24657) and surprise-gated writes (D-MEM) cover more of the same ground. Harrington et al. (2607.07847) supply the diagnosis. No search found a per-update router across context, store and adapter for skills and format shifts, nor a corroboration gate before weight writes; the gating work found protects external stores only. The remaining delta is real but incremental. It holds only if the gains on packets that are not facts and the noise gate show up at small scale. I could only read search snippets, not full papers, so Harrington's per-scenario details and Dual-Layer's experiments (benchmarks, noise tests) are unverified.

**What is taken and what is still new.** Dual-Layer Agentic Memory (2608.22215) handles facts with one write destination: whether to write to the external store, followed by SFT write-back. This plan (1) routes heterogeneous updates: skills, output-format conventions and noise as well as new and revised facts; (2) chooses among four destinations at write time, including a context-rule slot and a direct LoRA write, instead of store-first consolidation; (3) gates every weight write behind quarantine and corroboration, which none of the consolidation papers found (Dual-Layer, Dennis et al.) has; (4) reports a destination-by-type matrix with cost at 0.5-1.5B scale and an API arm. The Dual-Layer fact router becomes the key baseline, and the decisive test is the gain over it on packets that are not facts.

**Closest work:**

- [Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation (Li et al.)](https://arxiv.org/pdf/2608.22215) — arXiv 2608.22215, Aug 2026. Very high for facts. It has a fast write router with non-write, write-new and write-update labels, decided by zero-shot vs knowledge-supplied behavior (equivalent to the plan's s2 plus stubbornness), and slow SFT write-back from store to weights (the plan's Phase 2). It is limited to facts and the external store, with no skill or format destinations and no noise gate.
- [When Does Continual Learning Require Learning (Harrington et al.)](https://arxiv.org/pdf/2607.07847) — arXiv 2607.07847, Jul 2026. Diagnoses that the type of change determines weights vs scaffolding across 8 methods and 4 scenarios. It does not route per update.
- [D-MEM: surprise-gated memory routing](https://arxiv.org/pdf/2603.14597) — arXiv 2603.14597, 2026. A critic router uses surprise or reward-prediction error to decide whether inputs trigger memory evolution. The decision is external-memory only.
- [Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction (Dennis et al.)](https://arxiv.org/pdf/2605.24657) — arXiv 2605.24657, May 2026. Nightly LoRA consolidation beats context compaction (80% vs 37% retention); it has no per-update routing and no noise gate.
- [Understanding LoRA as Knowledge Memory: An Empirical Analysis](https://arxiv.org/pdf/2603.01097) — ICML 2026. Asks when LoRA memory should complement or replace RAG and ICL; it is a characterization of facts as knowledge, not a write router.
- [MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents](https://arxiv.org/html/2605.00356v1) — arXiv 2605.00356, 2026. A learned admission router for external memory; it never writes to weights.
- [Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory](https://arxiv.org/pdf/2606.25115) — arXiv 2606.25115, 2026. A provenance-penalized write scoring blocks poisoning, but only for the external store.

**Queries run:** router deciding whether new knowledge should go to retrieval memory or fine-tuning weights continual learning LLM (extended); Harrington 2026 continual learning LLM environmental change weights or scaffolding GEPA ACE Cartridges SDFT; "when to fine-tune" vs in-context vs RAG knowledge injection type-dependent memory write policy agent 2025 2026 arXiv; "Dual-Layer Agentic Memory" fast write routing slow consolidation; continual learning LLM route update to parametric or non-parametric memory based on knowledge vs skill, LoRA vs retrieval adaptive write 2026; "Understanding LoRA as Knowledge Memory" empirical analysis RAG complementary; memory write gate quarantine noisy or poisoned updates corroboration before consolidation into LLM weights continual learning; agent decides where to store new experience: prompt/system instructions, retrieval memory, or LoRA adapter weights; meta-controller memory placement (extended); "Beyond Inference-Only Deployment" weight-based consolidation cascading compaction; in-context learning gain predicts whether fine-tuning needed per example ... knowledge vs format vs skill; 2608.22215 write router write-new write-update cascade 1.7B 8B consolidation SFT benchmark knowledge updates noise

---

## 15. Depth, Not Time: Lineage Depth Predicts When Rewritten Agent Memory Rots, and Depth-Bounded Regeneration Prevents It

**Judge:** overall 4.5/10, tier C. Scores out of 5: novelty 2, low compute 2, impact 3, speed of first signal 3.
It addresses the doc's 2605.12978 lesson (rewrite-driven memory rot), and lineage depth as a schedule-independent state variable is a neat idea. But 2605.12978, 2609.25052 and Yang et al. already have the inverted-U and analytic error-dynamics models, and the plan relies on 8B models at 170 GPU-hours.

**Biggest weakness:** Depth may not collapse schedules onto one curve, and predicting the accuracy peak out of sample from telephone probes is ambitious. The compute exceeds the budget preference unless moved to API, and MemTrace or 2609.25052 may already index rot by rewrite count.

**One line.** Each insight in a procedural agent memory records how many LLM rewrite passes separate it from the raw episodes it came from. The claim is that utility depends on the consolidation schedule only through this depth distribution. Per-pass loss and corruption rates, measured on evidence-free 'telephone' probes, should then predict when accuracy peaks on ALFWorld and ARC. A cheap fix follows: regenerate any entry deeper than d* from raw episodes, or use O(log N) tree consolidation.

**Memory store.** External store: a consolidated text insight memory (cheatsheet or playbook) over an append-only raw-episode buffer. Every insight keeps provenance pointers to its source episodes and a lineage-depth counter.

**Task.** Procedural insight memory for sequential agents. (1) Hidden-Rule World (HRW, new, runs on CPU): a procedural text world with 64 latent conditional rules. About half are deliberately counter-to-commonsense, e.g. 'keys open doors only when the door is red'. Each episode reveals 1-3 rules, so every memory insight can be scored exactly as correct, missing, or corrupted (condition dropped, overgeneralized, invented, or reverted to the commonsense default). (2) ALFWorld text-only, 134-game valid_unseen split, with ReasoningBank/AWM-style insight memory. (3) A 100-task ARC-AGI-1 stream that re-creates the improve-plateau-decline setting of 2605.12978. (4) Game of 24 from the Dynamic Cheatsheet harness.

**Gap in the big picture.** This targets the document's flagship external-memory failure and its stated open problem. 2605.12978 shows that LLM-consolidated memory improves, plateaus, then falls below no memory, and that in-place rewriting causes the damage, but it does not say when or why quantitatively. The document names consolidation (what moves, and when) as the open problem. Recent analytic models do not answer this. 2609.25052 uses a per-model copy function to predict the direction of drift in append-only fact stores, and Yang et al. (cited there) model the steady-state occupancy of erroneous entries under replacement. Both are indexed by time or occupancy. None can tell apart two schedules that see identical experience but rewrite it a different number of times. None predicts when the inverted-U peaks on procedural memory, and none derives a regeneration schedule. Provenance DAGs (MemLineage) and raw-evidence capsules (DreamBench-SWE) supply the bookkeeping but no theory and no depth threshold.

**Hypothesis.** Holding experienced episodes fixed, the depth distribution of memory entries predicts procedural-memory utility across consolidation schedules (leave-one-schedule-out R^2 >= 0.7, at least 0.15 above time- and experience-indexed models). Per-pass loss and corruption rates measured only on evidence-free telephone probes predict the accuracy peak t* on ALFWorld and ARC within 25% of stream length.

**Method.**

Bookkeeping. Every insight stores provenance (source episode IDs) and lineage depth d. Raw-derived insights have d = 1, and any LLM pass that rewrites, merges or refines an entry sets d = 1 + the maximum depth of its parents. All memory versions are snapshotted, git-style.

Channel model. In each pass an insight stays correct with probability 1 - delta - gamma, is lost with probability delta, and is corrupted with probability gamma, split by type: overgeneralize, drop condition, invent, misgroup. Then P_correct(d) = (1 - delta - gamma)^d, and P_corrupt(d) is the matching two-state Markov expression. Memory utility is U = sum over insights of [b * P_correct(d_i) - lambda * P_corrupt(d_i)], minus a retrieval-competition term in memory size. With rule-discovery rate g(s) = g0 * exp(-rho * s), per-interaction rewriting (d = t - s) gives a closed-form inverted-U with peak t* and a no-memory crossover t_x. Batch-B (d = (t-s)/B), tree (d = ceil(log_b N)) and append-only (d = 1) schedules change only the map from s to d, so the model predicts that all schedules collapse onto one curve against depth.

Parameter measurement. delta and gamma come from 200 telephone passes per consolidator on HRW. A pass rewrites memory with the consolidator's own update prompt and either no evidence or one irrelevant episode, which separates pure drift from misgrouping. Rates are scored exactly against the rule set. 'Prior pull' measures the direction of corruption: the change in embedding similarity of an entry toward a memory the same LLM writes from task descriptions alone. On HRW this is checked exactly as reversion of counter-commonsense rules to their defaults.

Theory test. Model comparison against a time-only model, a Yang-style replacement-occupancy Markov model and a 2609.25052-style copy-function model, with leave-one-schedule-out fits. Then an out-of-sample prediction: t* and t_x for each of four consolidators on ALFWorld and ARC are pre-registered from HRW-measured parameters before those streams are run.

Remedies derived from the model. (a) Depth-bounded regeneration (DBR): when d > d*, re-derive the entry in one pass from its provenance episodes, where d* is the largest d with P_corrupt(d) <= epsilon. (b) Map-reduce tree consolidation with depth O(log N). (c) Tag-and-capture: a lesson is promoted only if it is independently re-derived from k >= 2 disjoint episode subsets, which cuts effective gamma by agreement. Remedies are compared at matched memory-operation tokens.

Secondary tool. Bisection over memory versions on a cached 40-task regression suite localizes harmful edits in O(log T) evaluations, validated on HRW where exact attribution is available. It is presented as an evaluation tool unless MemTrace and the 2605.12978 appendix turn out not to cover edit-level attribution.

**Datasets.**

- Hidden-Rule World (new, CPU procedural text environment; 64 latent conditional rules, ~half counter-commonsense; exact per-insight ground truth; to be released)
- ALFWorld text-only (github.com/alfworld/alfworld), 134-game valid_unseen split, 2 passes per stream
- ARC-AGI-1 (github.com/fchollet/ARC-AGI): 100-task stream (60 training-split + 40 evaluation-split tasks, fixed order, 3 seeds) re-creating the 2605.12978 decline protocol
- Game of 24 (task files from github.com/suzgunmirac/dynamic-cheatsheet)
- Regression suites: 40 cached past tasks per benchmark for version bisection

**Models / checkpoints.**

- Qwen/Qwen3-8B (actor and consolidator, served with vLLM)
- meta-llama/Llama-3.1-8B-Instruct (consolidator; a second open model family for varying delta/gamma)
- Qwen/Qwen3-1.7B (weak consolidator, expected high delta/gamma, widens the spread of parameters for the prediction test)
- GPT-5-mini via API (strong consolidator; ARC actor; budget <= $300)
- Qwen/Qwen3-Embedding-0.6B (prior-pull similarity and raw-episode retrieval; BAAI/bge-small-en-v1.5 as fallback)

**Baselines.**

- No memory
- Raw-trajectory retrieval (top-k episodes, no consolidation), reported strong by 2605.12978
- Append-only insight memory (depth 1 always)
- Dynamic Cheatsheet-Cumulative and DC-RetrievalSynthesis (official repo, MIT)
- ACE grow-and-refine with curator dedup (official repo ace-agent/ace, Apache-2.0)
- ReasoningBank-style distillation from successes and failures (google-research/reasoning-bank, re-targeted to ALFWorld/ARC)
- AWM-style online workflow induction (zorazrw/agent-workflow-memory, re-targeted)
- 2605.12978's recommendation, re-implemented from its description: raw episodes as first-class evidence plus gated consolidation
- Forced per-interaction rewrite and batch-B consolidation (B in {5, 20}) as schedule probes
- Theory baselines: time-indexed decay model, Yang-style replacement-occupancy Markov model, 2609.25052-style copy-function drift model

**Metrics.**

- Per-pass insight survival (1-delta) and corruption rate gamma by type (HRW, exact)
- Prior pull: change in similarity toward the description-only memory, and the reversion rate of counter-commonsense rules
- Task success / accuracy vs stream position; peak accuracy, final accuracy, area under the stream curve
- Leave-one-schedule-out R^2 and AIC of depth vs time vs occupancy vs copy-function models
- Absolute error in predicted t* and no-memory crossover t_x, as % of stream length (pre-registered)
- Spearman correlation of predicted vs observed t* across 4 consolidators x 2 benchmarks
- Memory-operation tokens (consolidation + regeneration) and memory size
- Harmful-edit localization: bisection precision/recall vs HRW ground truth; number of evaluations used
- Prior-pull AUROC for flagging harmful edits before the accuracy peak

**First experiment (go/no-go).**

Weeks 1-2, one GPU (80 GB, or a 48 GB L40S), about 20 GPU-hours.
(1) Build HRW in Python: 64 rules, an episode generator, and an exact insight scorer. Consolidators write insights as free text plus a structured mini-DSL 'rule' field. The scorer checks the DSL field exactly, and an LLM judge on free text is validated against it on 200 items (target kappa >= 0.8).
(2) Run telephone probes for Qwen3-8B, Llama-3.1-8B-Instruct and Qwen3-1.7B: 200 passes each, half evidence-free and half with one irrelevant episode. Estimate delta and gamma by type, with bootstrap confidence intervals, plus the counter-commonsense reversion rate.
(3) Feed one fixed 300-episode HRW stream (3 seeds) through 5 schedules with Qwen3-8B as actor and consolidator: per-interaction rewrite, batch-5, batch-20, tree, append-only. Regress per-checkpoint utility on depth-distribution features vs time and number of experiences.
GO if all three hold: (i) gamma >= 1% per pass for at least one model, and the delta/gamma confidence intervals differ across models; (ii) depth features reach leave-one-schedule-out R^2 >= 0.7 and beat time/experience by >= 0.15; (iii) per-interaction rewrite shows a peak followed by a decline on HRW while tree and append-only do not.
NO-GO if evidence-free corruption is < 0.5% per pass (rot then comes from misgrouping new evidence, not depth), or if depth adds < 0.05 R^2.

**Kill criterion.**

Drop the theory paper in any of these cases. (a) On HRW, schedules with identical experience do not collapse onto one curve against depth (depth adds < 0.05 held-out R^2 over time- or occupancy-indexed models). (b) Pre-registered t* predictions from HRW-measured delta/gamma miss by > 40% of stream length on both ALFWorld and ARC, or rank consolidators no better than chance (Spearman < 0.3). (c) The improve-plateau-decline curve cannot be reproduced with any of the four consolidators within 2 passes of ALFWorld or the 100-task ARC stream. (d) Reading 2609.25052, Yang et al., MemTrace or the 2605.12978 appendix shows a depth- or rewrite-count-indexed model already predicting peak timing. If only the theory fails while DBR and tree consolidation still beat 'raw + gated' on tokens at equal accuracy, downgrade to a short workshop paper on the remedies.

**Compute.** 170 GPU-hours over 14 weeks. One 80 GB GPU (A100 or H100) running vLLM with prefix caching; a 48 GB L40S also works for the 8B models. A multi-core CPU for HRW, the ALFWorld text engine and a linear-Gaussian simulation that checks the replace, accumulate and regenerate regimes. No training. Budget breakdown:
- HRW probes and streams: ~28 GPU-h
- ALFWorld streams (10 policies x 3 seeds x 2 consolidators): ~60 GPU-h
- ARC with Qwen3-8B: ~45 GPU-h
- Game of 24: ~10 GPU-h
- Bisection regression evaluations: ~10 GPU-h
- Slack: ~17 GPU-h
API spend is <= $300 (GPT-5-mini as ARC actor and consolidator, plus judge calls).

**Risks.**

1. The decline may not reproduce. 2605.12978 showed it with a frontier model (GPT-5.4) on ARC. 8B consolidators may degrade too fast (no peak) or ARC accuracy may sit near the floor, giving no signal. Mitigations: GPT-5-mini for ARC, an easier training-split subset, and ALFWorld as the main open-model benchmark.
2. Two parameters may be too crude. Corruption depends on content type, memory length budget and size, so delta and gamma may need to be per-type or size-conditioned. That weakens the 'sufficient statistic' claim.
3. HRW rules may not resemble real procedural insights. Mitigations: the counter-commonsense rule design, and validating parameter transfer through the ALFWorld/ARC prediction test.
4. ACE's delta updates and dedup keep depth low by construction. The model must account for refine and merge steps that raise depth, or ACE becomes a trivial 'good' point.
5. The headline could collapse into 'do not rewrite in place', which 2605.12978 already recommends. The quantitative prediction and d* must carry the paper.
6. Unverified overlap. 2609.25052, Yang et al., MemTrace and the 2605.12978 appendix were known only from search snippets: the search budget ran out and arxiv.org/huggingface.co failed DNS in this session. Read them in week 1.
7. Agent stochasticity makes regression deltas and bisection noisy. Mitigation: 3 seeds, cached suites and greedy decoding for regression checks.
8. Prior pull may also fire on harmless compression.
Checkpoints and repos verified this session: ALFWorld, ARC-AGI, Dynamic Cheatsheet, ACE, AWM and ReasoningBank repos exist. The Hugging Face model pages could not be opened, so model names follow standard HF IDs.

**Target venue.** ICML 2027 (deadline about late January 2027, which fits the 14-week plan); fallback COLM 2027 or the NeurIPS 2027 main track

**First-round prior-work verdict: partially_novel.** 2605.12978 already reports the inverted-U, the edit-type taxonomy, and the finding that rewriting is worse than accumulating. 2609.25052 and Yang et al. already have analytic error-dynamics models built on per-model primitives, so 'first law of memory rot' cannot be claimed. The defensible delta is narrower: lineage depth as the schedule-independent state variable, tested by holding experience fixed and varying schedules; out-of-sample prediction of peak timing on procedural benchmarks from evidence-free probes; and remedies derived from the model and compared at matched tokens. Edit-level bisection is presented as an evaluation tool, because MemTrace (2605.28732) could not be checked. This session's web-search budget was exhausted and arxiv.org was unreachable, so no overlap judgment rests on a full-text read.

**Closest work found in the first round:**

- [Useful Memories Become Faulty When Continuously Updated by LLMs (Zhang et al.)](https://arxiv.org/abs/2605.12978) — arXiv 2605.12978, May 2026. The paper this idea sets out to explain. Going by search summaries, it reports the rise-then-fall curve that drops below no memory, on ARC-AGI Stream with GPT-5.4. It names three mechanisms: misgrouping, stripped applicability conditions (overgeneralization) and overfitting. It ablates the update operator at step 400: forced consolidation 43.2%, append-only 70.0%, raw trajectories 76.6%, add/delete 64.0%. From this it argues that in-place rewriting adds damage beyond abstraction loss, and it recommends raw episodes as first-class evidence plus gated consolidation. That already covers the qualitative edit-type taxonomy and the 'rewrite vs accumulate' finding. It does not appear to have a per-pass channel model, a depth variable, predicted t*, or bisection. I could not open the full text, so appendix-level edit analysis is still uncertain.
- [Self-Cleaning and Captured Anyway: One Measured Primitive for Error in a Store an Agent Writes to Itself, and What a Falling Score Actually Measures (Chen, Chen, Lin, Vong)](https://arxiv.org/abs/2609.25052) — arXiv 2609.25052, Sep 2026. The most threatening one on the theory side. It is an analytic model of error dynamics in a store an agent writes to itself, driven by a single measured per-model primitive with no fitted parameters, the copy function gamma(phi). sign(gamma - gamma_crit) predicts the direction of drift in 353 of 360 runs on Wikidata facts plus synthetic runs, and the paper also discusses what a falling score actually measures. This occupies the 'measure a per-LLM channel parameter, then predict the memory decline analytically' slot. Differences: its store is append-only with fact contamination, it finds bimodal edges rather than an inverted-U, and it does not use rewrite depth or procedural insights. Known only from search snippets.
- [Yang et al., steady-state occupancy of erroneous coordinates in self-evolving agent memory (cited by 2609.25052)](https://arxiv.org/html/2609.25052v1) — 2026 (exact venue and title not confirmed). 2609.25052 describes it as a coordinate-transition (Markov) model that characterizes the steady-state occupancy of erroneous memory entries in a self-evolving agent memory with replacement. That is close to the idea's per-pass keep/corrupt/add channel and its replace regime. I could not retrieve the actual paper, so how close it is remains uncertain. The URL given is the citing paper.
- [MemLineage: Lineage-Guided Enforcement for LLM Agent Memory](https://arxiv.org/abs/2605.14421) — arXiv 2605.14421, May 2026. Attaches provenance and an LLM-mediated derivation lineage, a weighted derivation DAG, to every memory entry. That is the bookkeeping the idea needs for lineage depth. Its use is security and chain of custody, not predicting utility decay or scheduling regeneration.
- [DreamBench-SWE: A Multi-Session Memory-Hygiene Benchmark for Software Agents](https://arxiv.org/abs/2608.20664) — arXiv 2608.20664, Aug 2026. Keeps verbatim raw-evidence capsules per episode that are exempt from repair, plus a provenance gate that marks derived memories active or requires_review depending on whether raw episodes support them. That makes derived memory suppressible and re-derivable from raw evidence, partly overlapping remedy (a), regeneration from raw episodes. It has no depth threshold or theory.
- [Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory (Kang et al.)](https://arxiv.org/abs/2607.17545) — arXiv 2607.17545, Jul 2026. Studies when to consolidate versus retain raw records, and finds it depends on budget (+48% under tight budgets on LongMemEval). It also lists the corruption types that consolidation introduces: omitted details, blurred order, dropped corrections, unsupported content. This overlaps the 'consolidation scheduling' framing but uses a budget axis, not depth.
- [MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems](https://arxiv.org/abs/2605.28732) — arXiv 2605.28732, May 2026. From the title alone, it does error tracing and attribution in LLM memory systems, which may overlap the idea's edit-level attribution and bisection. I could not check the details. This is a possible threat to hypothesis (3).
- [LLM as a Broken Telephone: Iterative Generation Distorts Information (Mohamed et al.)](https://arxiv.org/abs/2502.20258) — arXiv 2502.20258, 2025. Measures cumulative distortion under iterated LLM regeneration in translation chains. This is the telephone-probe methodology outside agent memory. Together with Perez et al. 2024 (attractors in iterated transmission), it covers the 'prior pull / attractor' direction-of-drift idea in general text.
- [SSGM (semantic drift governance for evolving memory)](https://arxiv.org/html/2603.11768) — arXiv 2603.11768, Mar 2026. Names semantic drift during consolidation updates as a key failure point, and notes that errors compound in a feedback loop. It is a governance framework, not a quantitative predictor.
- [Knowledge Objects (compaction erodes constraints)](https://arxiv.org/html/2603.17781v1) — arXiv 2603.17781, Mar 2026. Reports that summarization destroys 60% of facts and that cascading compaction erodes 54% of constraints. That is empirical evidence of per-pass loss compounding with depth, but there is no model or schedule.

---

## 16. How Much Replay Does a Pretrained ViT Need? Replay-Demand Laws and Drift Sensors for Compute-Bound Continual Learning

*First-round title: How Much Replay Does a Pretrained ViT Need, and When? Replay-demand laws and a KL thermostat for compute-budgeted continual learning*

**Judge:** overall 4.5/10, tier C. Scores out of 5: novelty 2, low compute 3, impact 2, speed of first signal 4.
The recheck removed the thermostat: 2602.22479 already runs a PI forgetting-gap replay controller, and Bethune et al. and Hickok cover replay-fraction laws and replay saving. What remains is a vision measurement paper on r*(T) and a proxy race. It is solid but incremental, and far from the doc's cross-store consolidation and 3D interests.

**Biggest weakness:** Pretrained ViT with LoRA may show almost no forgetting (kill criterion 1), and 'measurement only' papers on class-incremental benchmarks are a hard sell. The first-pass compute is optimistic at Mammoth scale.

**One line.** Keep every training image in the buffer. Measure the smallest replay fraction that holds forgetting under epsilon as the number of tasks grows, and how that depends on how many parameters are allowed to change (LoRA rank vs full fine-tuning). Test whether KL drift against write-time logits stored on old data predicts which classes will be forgotten better than parameter distance or RL's-Razor new-task KL. Then use the best sensor in a closed-loop thermostat that spends replay compute only when and where drift appears.

**Memory store.** Data buffer (raw samples plus write-time logits, unlimited size). It controls how much replay goes into the trainable weights (LoRA adapters or the full backbone) versus head-only distillation, and when.

**Task.** Replay under a compute budget, on two kinds of stream. (1) Class-incremental: Split CIFAR-100 (2 classes per task, 50 tasks; 10-task split for literature comparison) and ImageNet-R (5 classes per task, 40 tasks; standard 10-task split). (2) Domain-incremental natural temporal drift: CLEAR-100 yearly buckets with the train-on-t/test-on-t+1 protocol. Backbones are pretrained ViTs adapted by LoRA or full fine-tuning, with an unlimited buffer. Compute is the binding budget, not storage.

**Gap in the big picture.** The doc's lesson 1 (Cho et al. 2502.07274) says replay is the bar to beat and compute, not storage, is the real budget. It also names a data-buffer failure mode: replay compute grows with history. Nobody has measured whether replay need actually grows with history for pretrained backbones, or how that depends on how many parameters are allowed to change. Lesson 2 (RL's Razor) says forgetting tracks the KL shift an update causes. For vision continual learning it is untested whether that KL should be measured on old data (which needs a buffer) or on the new task's data (which needs none), and whether it beats parameter-space proxies. Fixed-ratio replay ignores both questions: it pays replay compute even while old classes are not drifting.

**Hypothesis.** With an unlimited buffer, the smallest replay fraction r*(T, epsilon) that keeps average forgetting under epsilon grows sublinearly with the number of tasks T for LoRA-adapted pretrained ViTs, with constant or log fits beating a linear fit by delta-BIC > 10. It grows near-linearly for full fine-tuning of ResNet-18 from scratch. In addition, old-data sentinel KL predicts per-class forgetting with Spearman >= 0.6, at least 0.1 above both parameter distance and the RL's-Razor new-task KL.

**Method.**

(1) Sensor. Mammoth's DER buffer stores every image with its write-time logits. A stratified sentinel set (the larger of 1% of the buffer and 5 items per class) rotates through the buffer and is forward-passed every 50 steps. This gives a per-class KL(p_write || p_now), which splits exactly into two terms: within-old-class drift, KL(p_write || q_now), where q_now is p_now renormalized over the classes seen at write time; and a recency-leak term, -log(1 - m_now), where m_now is the probability mass on classes added since write time. The split shows whether drift sits in the classifier head or in the features.

(2) Proxy race. At every task boundary, the sensor is scored against six alternatives: parameter L2 and Fisher-weighted distance from the task-start weights, buffer-vs-new gradient cosine, learning-speed scores, the RL's-Razor proxy (KL from the task-start model on the new task's own data, which needs no old data), and KL on a fixed public probe set (1k Places365 val images). The score is per-class Spearman and AUROC against the accuracy drop realized at the end of the next task and at the end of the stream.

(3) Replay-demand law. Classes per task stay fixed so that only history length varies. We run constant replay fractions rho in {0, .05, .1, .2, .35, .5} and read off r*(T, epsilon) = the smallest rho whose average forgetting stays at or below epsilon at every prefix of length T. This is crossed with the size of the trainable parameter set: ViT-S LoRA r=4/16/64, ViT-S full fine-tuning, and ResNet-18 from scratch. Constant, log, power and linear forms are fitted and compared by BIC with bootstrap confidence intervals. A per-task oracle schedule, found by branching from task-boundary checkpoints, measures the headroom any adaptive policy could have.

(4) Thermostat. A PI controller sets each minibatch's replay fraction: rho_k = clip(rho_0 + Kp*e_k + Ki*sum(e), 0, 0.5), where e_k = EMA(class-mean sentinel KL) - epsilon, with anti-windup. Replay classes are drawn with probability proportional to max(KL_c - epsilon, 0) plus a small floor. The sentinel forward passes are reused for head-only distillation toward the write-time logits, which addresses the leak term at no extra compute. Gains are tuned once on CIFAR-100/10 with ViT-S and then frozen for every other stream.

(5) Equal-FLOP evaluation. Every method is compared at matched total FLOPs. These include training forward and backward passes, sensor forwards and MIR's virtual-update passes. Pareto curves sweep epsilon. The 20-step probe forecast from the original idea is demoted to a warm-start ablation, because it largely duplicates MIR and Jin & Ren.

**Datasets.**

- Split CIFAR-100: 2 classes x 50 tasks and 10 tasks. Mammoth seq-cifar100-224 for ViTs; 32px seq-cifar100 for ResNet-18.
- ImageNet-R (200 classes): 5 classes x 40 tasks and the standard 10-task split (Mammoth seq-imagenet-r).
- CLEAR-100: YFCC100M images bucketed by year 2004-2014, with bucket 0 kept for pretraining, built with github.com/linzhiqiu/continual-learning; streaming train-on-t/test-on-t+1 protocol.
- Places365 val, 1k-image subset: unlabeled probe set for the data-free KL proxy.
- Optional stretch: DomainNet, 6 domains subsampled to 50 images per class per domain, as a second domain-incremental stream.

**Models / checkpoints.**

- timm vit_base_patch16_224.augreg_in21k (ViT-B/16, ImageNet-21k), adapted with LoRA r=16 on qkv and MLP via Mammoth backbone/utils/lora_utils.py
- timm vit_base_patch16_clip_224.openai (CLIP ViT-B/16), a second pretrained regime with LoRA r=16; vit_base_patch14_dinov2.lvd142m as an alternative
- timm vit_small_patch16_224.augreg_in21k (ViT-S/16): LoRA r=4, 16 and 64, plus full fine-tuning, for the trainable-capacity axis
- ResNet-18 from scratch (Mammoth default backbone): non-pretrained contrast for the r*(T) law
- All four timm tags were checked against timm 1.0.15 pretrained configs

**Baselines.**

- No replay (sequential fine-tuning)
- ER at fixed replay fractions rho in {0.05, 0.1, 0.2, 0.35, 0.5} with an unlimited buffer, the Cho et al. 2502.07274 setting
- Best fixed rho in hindsight: an oracle constant ratio taken from the law grid, the key bar for any adaptive policy
- Per-task oracle replay schedule found by checkpoint branching (headroom upper bound)
- DER++ (Mammoth derpp); thermostat applied on top of both ER and DER++
- MIR (Aljundi et al. 2019), ported from github.com/RaptorMai/online-continual-learning, with virtual-update FLOPs counted
- Loss-triggered replay: the same PI controller driven by sentinel cross-entropy instead of KL (sensor ablation)
- Open-loop update-magnitude-triggered schedule in the style of FOREVER (2601.03938), reimplemented after the paper is verified
- Periodic replay every k steps at matched FLOPs
- Adaptive Memory Replay bandit (Smith et al. 2024), simplified reimplementation over class clusters
- SLCA and RanPAC (Mammoth slca/ranpac): strong no-replay references for pretrained ViTs

**Metrics.**

- Final average accuracy A_T and average forgetting F_T; backward transfer
- CLEAR-100 next-bucket accuracy (forward transfer) and in-bucket accuracy
- Replay FLOPs and total training FLOPs (analytic plus fvcore), and wall-clock time; area under the accuracy-vs-FLOPs Pareto curve
- Replay saved at equal accuracy relative to the best fixed rho in hindsight
- r*(T, epsilon) curves with bootstrap 95% confidence intervals; BIC of constant, log, power and linear fits; growth exponent as a function of LoRA rank
- Per-class Spearman and Kendall correlation, and AUROC for detecting a >5-point accuracy drop, for each drift proxy
- Share of sentinel KL due to the recency-leak term vs within-old drift, per stream and backbone
- Controller diagnostics: rho_k traces, KL overshoot above epsilon, sensitivity to epsilon and to Kp/Ki

**First experiment (go/no-go).**

Two weeks on one GPU, about 8 GPU-hours.

Days 1-2:
- Read FOREVER (2601.03938) and MSSR (2603.09892) to check for closed-loop control of the replay fraction or a fitted law of replay versus task count. They could not be verified during planning.
- Set up Mammoth seq-imagenet-r and seq-cifar100-224 with ViT-B/16 in21k LoRA r=16 and with ViT-S/16 full fine-tuning. Use 5 and 3 epochs per task respectively, down from Mammoth's defaults of 50 and 20.
- Add logging hooks for every proxy. Write-time logits are already stored by Mammoth's DER buffer.

Days 3-7, runs:
- ImageNet-R/10 and CIFAR-100/10 x {ViT-B LoRA, ViT-S full fine-tuning} x rho in {0, 0.1, 0.3} x 2 seeds: 24 runs, about 5 GPU-hours.
- ViT-S full fine-tuning on the 40-task ImageNet-R stream across the six-value rho grid, 1 seed: about 1 GPU-hour, giving a first r*(T) curve.
- Per-task oracle schedule on ImageNet-R/10 with ViT-S, by branching over 4 rho values at each boundary: about 1 GPU-hour.

Days 8-10, analysis:
- Per-class Spearman and AUROC of each proxy against realized forgetting.
- Split of sentinel KL into the leak term and within-old drift.
- Shape of the first r*(T) curve.
- Replay saved by the oracle schedule against the best constant rho.

Verdict:
- GO if all of the following hold: (a) no-replay forgetting is at least 5 points for some pretrained configuration; (b) sentinel KL reaches Spearman >= 0.6 and beats parameter distance by >= 0.1; (c) the oracle schedule saves >= 25% of replay FLOPs at equal accuracy.
- Measurement-only paper (no thermostat) if (a) and (b) hold but (c) fails.

**Kill criterion.**

Drop the project in any of these cases:
- No-replay average forgetting is under 3 points for every pretrained configuration, including full fine-tuning of ViT-S on 40-task ImageNet-R. There is then nothing for replay to manage.
- Sentinel KL's per-class Spearman with realized forgetting is under 0.4 and no better than parameter distance, so H1 fails and the controller has no reliable sensor.
- Day-1 reading shows that FOREVER or MSSR already does both closed-loop, drift-driven replay-fraction control and fits replay versus task count.

Drop only the thermostat and keep the measurement paper if either holds:
- The per-task oracle schedule saves under 25% of replay FLOPs against the best fixed rho in hindsight. That would replicate Prabhu et al.'s finding that equal-compute ER is hard to beat.
- In the full comparison, the frozen-gain thermostat fails to beat the best fixed rho by 1 point or 25% of replay FLOPs on at least two of the four streams.

**Compute.** 150 GPU-hours over 12 weeks. One A100 80GB, or one L40S 48GB, or one RTX 4090 24GB using bf16 and batch 64. CPU only for law fitting and bootstrap. Budget breakdown:
- Week-1 proxy race and headroom check: 8 GPU-hours.
- Replay-demand grid: 144 ViT-S runs at about 0.15 h, 48 ViT-B LoRA runs at about 0.4 h, and 12 ResNet-18 runs, about 45 GPU-hours.
- Equal-FLOP method comparison: 10 methods x 4 streams x 2 backbones x 3 seeds, reusing the fixed-ER runs, plus the epsilon Pareto sweep and SLCA/RanPAC references: about 60 GPU-hours.
- Ablations (sensor type, sentinel size, P vs PI vs bang-bang, class allocation, warm-start probe): about 20 GPU-hours.
- Debugging, plus a one-seed check that the method ranking holds under Mammoth's full 50-epoch ImageNet-R schedule: about 17 GPU-hours.

**Risks.**

(1) Scoop risk. FOREVER, MSSR, or an LLM data-injection scaling law may already contain the controller or the law, and none could be verified during planning. Mitigation: read them on day 1 and reposition before running anything.

(2) The policy may have no headroom. Prabhu et al. found that at equal compute, uniform ER is hard to beat, so the thermostat may only match the best fixed ratio. Mitigation: the week-1 oracle-schedule check, with a measurement-only paper as the fallback.

(3) Pretrained LoRA ViTs may barely forget, especially on CLEAR-100, which would give r* close to 0. Full fine-tuning of ViT-S and ResNet-18 from scratch are included so that some regimes forget meaningfully.

(4) Class-incremental drift may be dominated by head recency bias, which logit adjustment fixes without replay. The leak/within-old split makes this visible, and head-only distillation from reused sentinel passes is part of the method, so this cannot be hidden.

(5) Sentinels may be too few. At 1% of ImageNet-R that is about 1 per class, so a per-class floor of 5 is enforced; at ViT-B that costs about 1 s per check.

(6) Fits over T <= 50 cannot separate log from sqrt growth, so claims are limited to sublinear vs linear, with BIC and bootstrap.

(7) Epochs per task are reduced from Mammoth's defaults. A one-seed check that the method ranking holds under the full schedule guards against artefacts of short training.

(8) Epsilon and the PI gains are extra hyperparameters. Gains are tuned once on CIFAR-100/10 and frozen, and sensitivity is reported.

**Target venue.** ICML 2027 (late-January deadline) or CoLLAs 2027; TMLR if it ends up a measurement-only paper

**First-round prior-work verdict: partially_novel.** Every component has prior art:
- Write-time logits: DER.
- Virtual-update interference: MIR.
- Learned or adaptive replay scheduling: Klasson et al.; Smith et al. 2024; FOREVER 2026.
- Forecasting forgotten examples: Jin & Ren.
- KL as a forgetting law: RL's Razor.
- Equal-compute evaluation: Prabhu et al.; Cho et al.

What remains defensible:
- A measured scaling law r*(T, epsilon) for minimal replay, with an unlimited buffer, crossed with the size of the trainable parameter set (LoRA rank vs full fine-tuning vs training from scratch).
- A head-to-head race between old-data KL, data-free new-task KL and parameter-space drift proxies, with an exact split of drift into recency leak and within-old drift.
- A closed-loop PI controller over how much to replay, rather than which samples, evaluated against a best-in-hindsight fixed ratio and a per-task oracle schedule.

Caveat: no live search was possible during planning. The web-search budget was exhausted and arxiv.org did not resolve. FOREVER (2601.03938), MSSR (2603.09892) and any LLM data-injection scaling law are the likeliest scoops and must be read on day 1. Only the codebases (Mammoth, PILOT, CLEAR, RaptorMai OCL) and the timm checkpoint tags were verified.

**Closest work found in the first round:**

- [Learn the Time to Learn: Replay Scheduling in Continual Learning (Klasson, Kjellstrom, Zhang)](https://arxiv.org/abs/2209.08660) — TMLR 2023 (arXiv 2022). Learns which old tasks to replay at each step under a fixed replay budget, using MCTS and later an RL policy. This is the closest prior work on deciding when and which tasks to replay. It differs from the idea in three ways: it is not a closed-loop KL sensor, it fixes the budget size rather than deciding how much to replay, and it ran on small CNN benchmarks. NOT VERIFIED: recalled from memory because the network was blocked. The arXiv ID is my best recollection and may be wrong.
- [Adaptive Memory Replay for Continual Learning (Smith et al.)](https://arxiv.org/abs/2404.12526) — CVPR Workshops 2024 / arXiv 2024. A multi-armed bandit chooses which past-data clusters to replay, based on loss, for continual pretraining of foundation models under compute limits. It already covers adaptive, compute-aware replay for pretrained models. It does not use a KL sensor, a PI control loop or an adaptive replay fraction. NOT VERIFIED: recalled from memory. The idea itself also cites it. The arXiv ID is from memory.
- [Online Continual Learning with Maximally Interfered Retrieval (MIR, Aljundi et al.)](https://arxiv.org/abs/1908.04742) — NeurIPS 2019. Runs a virtual update and measures how much each buffer sample's loss would rise, then replays the most interfered samples. This is the same operation as the idea's 20-step probe and drift-weighted replay draw, at per-step granularity. The difference is that MIR picks which samples to replay, not how much. Outside the 2022-2026 window, but it is the mechanistic ancestor. Recalled from memory and not opened.
- [FOREVER (arXiv 2601.03938), as described in the idea's own closest_known_work](https://arxiv.org/abs/2601.03938) — arXiv 2026. According to the idea text, it sets the replay schedule open-loop from optimizer-update magnitude, for LLMs. This is the most direct 2026 threat to the 'when to replay' claim. I could not fetch it (arxiv.org blocked), so I could not confirm whether it also adapts the replay amount in closed loop or uses an output-space signal.
- [MSSR (arXiv 2603.09892), as cited in the idea](https://arxiv.org/abs/2603.09892) — arXiv 2026. The idea lists it as close prior work on replay scheduling. I could not fetch it, so its mechanism is unverified. It may already use a drift or forgetting signal to set replay, which would directly threaten the controller contribution.
- [Memory Is Not the Bottleneck: Cost-Efficient Continual Learning via Weight Space Consolidation (Cho et al.)](https://arxiv.org/abs/2502.07274) — arXiv 2025. Recasts continual learning with an unlimited buffer and compute as the binding budget, and shows plain replay is strong for pretrained models. This supplies the idea's framing and baseline but not its controller. Its link and title are confirmed only through the user's big-picture notes; I did not open it.
- [Computationally Budgeted Continual Learning: What Does Matter? (Prabhu et al.)](https://arxiv.org/abs/2303.11165) — CVPR 2023. Compares continual learning methods, including sampling strategies and distillation, under equal-compute budgets, and finds that at equal compute, simple ER with uniform sampling is hard to beat. This is the idea's equal-FLOP evaluation protocol. It also implies that sophisticated sampling rarely pays off, which threatens H3. Recalled from memory; not opened.
- [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement (Jin & Ren)](https://arxiv.org/abs/2402.01865) — ICML 2024. Forecasts which upstream examples an update will make the model forget, from logit or representation changes, and uses the forecast to choose what to replay. This overlaps the idea's pre-hoc probe forecast (part 2) for language models. Recalled from memory, and the arXiv ID is uncertain.
- [RL's Razor: Why Online Reinforcement Learning Forgets Less (Shenfeld et al.)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html) — ICLR 2026. Shows that forgetting is predicted by KL from the base model, measured on the new task's distribution. H1 transfers this law to vision continual learning, measured on old-data sentinels. That empirical claim is partly anticipated, but the use of KL as a control signal is not. Citation taken from the user's big-picture notes; the page could not be fetched.

### Second-round re-check (live search)

**Revised one line.** With an unlimited buffer, measure how the smallest replay fraction that keeps forgetting under epsilon grows with the number of tasks, for each size of trainable set (LoRA rank vs full fine-tuning vs from scratch). Race old-data write-time-logit KL against new-task KL (RL's Razor) and parameter-space proxies as per-class forgetting predictors. Use a per-task oracle schedule to test whether any adaptive replay controller, including existing PI and RL ones, has headroom over the best fixed ratio at equal FLOPs.

**Verdict: partially_novel; recommendation: pursue_with_pivot.**

**What is taken and what is still new.** All cited IDs resolve to the papers the plan claims: FOREVER 2601.03938 (ACL 2026), MSSR 2603.09892, Klasson 2209.08660, Smith 2404.12526, Jin & Ren 2402.01865, and Cho 2502.07274, which is now 'Forget Forgetting', ICLR 2026.

The search refutes the plan's claim that 'prior schedulers are open-loop'. Two works already close the loop:
- arXiv 2602.22479 (Thalamically Routed Cortical Columns, 2026) runs a PI-style replay controller. It measures the rise in past-task log-perplexity, smooths it with an EMA, takes the gap above a target as the error, keeps an integral state, and adjusts replay strength online.
- 'Closed-loop Control for Online Continual Learning' uses a held-out test memory as a feedback signal and an RL policy to set replay hyperparameters.

FOREVER already adapts both when and how strongly to replay, from update magnitude. MSSR adapts replay sparsity per sample.

The KL thermostat is therefore not a novel mechanism. What is left of it is the sensor (KL against write-time logits, split into recency leak and within-old-class drift) and its use in vision class-incremental learning on pretrained ViTs.

Bethune et al. (ICML 2025) already fit scaling laws of forgetting against the replay/injection fraction, for LLM fine-tuning. Hickok (2505.12512) already studies cutting the replay ratio with LoRA plus consolidation and reports replay saved.

What survives, with no direct match in 12 searches:
- (a) r*(T, epsilon): how the minimal replay fraction grows with the number of tasks, crossed with how many parameters may change (LoRA rank vs full fine-tuning vs a from-scratch ResNet), in vision continual learning with an unlimited buffer.
- (b) A per-class race between forgetting proxies: old-data write-time-logit KL vs RL's-Razor new-task KL vs parameter/Fisher distance, with the exact leak/within-old split.
- (c) An honest equal-FLOP test of whether any adaptive replay policy has headroom over the best fixed ratio in hindsight, using a per-task oracle schedule.

**Required revisions.** Drop these claims:
- That prior replay schedulers are open-loop.
- That a closed-loop PI controller over the replay fraction is a contribution. arXiv 2602.22479 already runs a PI-style, forgetting-gap-driven replay-strength controller, and 'Closed-loop Control for Online Continual Learning' already adapts replay online from test-memory feedback.

Recast the thermostat as an instantiation of these controllers with a better sensor, or as a probe of policy headroom. Its contribution then rests on the sensor comparison: the same PI loop driven by KL vs cross-entropy vs update magnitude.

Cite and position against:
- Bethune et al. ICML 2025: a replay-fraction scaling law, but for one LLM fine-tune with no T axis. State that r*(T) adds the number-of-tasks and trainable-capacity axes, in vision.
- Hickok 2505.12512: replay-ratio reduction plus consolidation, and 'replay saved' accounting.

Use the Jin & Ren follow-up 2406.14026 (low-rank forgetting matrices) as an extra predictor in the proxy race.

Correct the citation: Cho et al. 2502.07274 is now titled 'Forget Forgetting: Continual Learning in a World of Abundant Memory' (ICLR 2026). Note its finding that abundant replay hurts plasticity, which r*(T) curves should also report as new-task accuracy.

The day-1 kill check on FOREVER and MSSR is resolved:
- Neither fits replay against T.
- Neither uses an output-space sensor on old data.
- Both are for LLMs.
- FOREVER does modulate replay intensity open-loop. Keep it as a baseline.

The headline is now the measurement paper. The thermostat becomes a secondary section, or goes to the appendix if the oracle headroom is under 25%.

**Baselines to add.**

- Forgetting-gap PI replay controller of arXiv 2602.22479, ported: the same PI loop driven by old-task loss gap, as the direct closed-loop competitor
- Closed-loop Control for Online Continual Learning (RL-tuned replay hyperparameters from test-memory feedback)
- FOREVER (2601.03938) update-magnitude scheduler plus intensity-aware replay regularizer, ported to ViT
- MSSR (2603.09892) memory-strength-decay sample scheduler at matched replay ratio
- Hickok 2505.12512: reduced replay ratio plus post-task consolidation phase (with LoRA)
- Bethune et al. 2502.06042 functional form, fitted as an alternative to the plan's constant/log/power/linear family for r*(T)
- Jin & Ren low-rank forgetting-matrix predictor (2406.14026) as an extra entrant in the proxy race

**Closest work (second round):**

- [Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns](https://arxiv.org/pdf/2602.22479) — arXiv 2026 (2602.22479). The most direct scoop of the thermostat. According to the search snippet, it has a replay controller that tracks how much past-task log-perplexity has risen since each task ended, smooths that with a moving average, and defines the controller error as the amount by which the smoothed gap exceeds a target. It keeps a PI-style integral state and adjusts replay strength online. That is the same closed-loop structure as the plan: a forgetting sensor on old data, a target epsilon and a PI integral on replay strength. It differs in being for LLMs, in using a loss/perplexity sensor rather than KL to write-time logits, in having no per-class allocation, and in fitting no law.
- [Closed-loop Control for Online Continual Learning](https://astro.paperswithcode.com/paper/closed-loop-control-for-online-continual) — (venue/year not shown in snippet; paperswithcode listing). Gets real-time feedback on forgetting from an extra test memory, which plays the same role as the plan's sentinels. It uses RL to adjust replay hyperparameters online during the stream. This is a closed-loop replay controller for online continual learning. It differs in using an RL policy rather than a PI controller and in not using a KL sensor.
- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](https://arxiv.org/abs/2601.03938) — ACL 2026 (arXiv 2601.03938). Confirmed to exist. A forgetting-curve scheduler decides when to replay, using a model-centric clock (optimizer-update magnitude) instead of step counts. An intensity-aware regularizer, scaled by a parameter-update instability ratio, decides how strongly to replay. So it covers both when and how much, but open-loop: the signal comes from parameter space with no measurement on old data. LLMs only (Qwen3-0.6B to 13B). It is an update-magnitude baseline the plan must beat.
- [MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning](https://arxiv.org/pdf/2603.09892) — arXiv 2026 (2603.09892). Confirmed to exist. Gives each sample an Ebbinghaus-style memory strength that rises when it is replayed and decays over time. Replay gets sparser as memories stabilize, and weak samples get priority, at a given replay ratio, with LoRA. The schedule is adaptive but model-free: memory strength is modeled, not measured from model outputs. It fits no replay law versus task count. LLMs only.
- [Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection (Bethune et al.)](https://arxiv.org/pdf/2502.06042) — ICML 2025 (arXiv 2502.06042). Fits scaling laws of forgetting against the injected-replay fraction, model size and target-data size, and finds about 1% injection prevents forgetting. This is the replay-fraction scaling law for a single fine-tuning step in LLMs. It does not vary the number of tasks T, the size of the trainable set (LoRA rank vs full fine-tuning) or vision. It anticipates the form of H2, so the plan must cite it and set itself apart.
- [Scalable Strategies for Continual Learning with Replay (Hickok)](https://arxiv.org/pdf/2505.12512) — arXiv 2025 (2505.12512). Studies reducing the per-batch replay ratio (optionally with LoRA) and adding a post-task consolidation phase. Needs up to 55% fewer replay samples for a given performance target, and reports total replay usage relative to 1:1. This directly overlaps 'replay saved at equal accuracy' and the ratio-vs-LoRA axis, but it fits no r*(T) law and has no closed-loop sensor.
- [Learn the Time to Learn: Replay Scheduling in Continual Learning (Klasson, Kjellstrom, Zhang)](https://arxiv.org/abs/2209.08660) — TMLR 2023 (arXiv 2209.08660). Confirmed to exist. Uses MCTS and then an RL policy to choose which tasks to replay at each step under a processing-time budget. It is prior art on learned when/what replay scheduling and its oracle search, and matches the plan's per-task oracle schedule by checkpoint branching.
- [Adaptive Memory Replay for Continual Learning (Smith et al.)](https://arxiv.org/abs/2404.12526) — CVPRW 2024 (arXiv 2404.12526). Confirmed to exist. A bandit with Boltzmann sampling adjusts each past task's share of replay by its propensity to be forgotten. It assumes full data access and targets training efficiency. Prior art for forgetting-proportional class/cluster allocation.
- [Memory Is Not the Bottleneck / Forget Forgetting: Continual Learning in a World of Abundant Memory (Cho et al.)](https://arxiv.org/abs/2502.07274v3) — ICLR 2026 (arXiv 2502.07274). Confirmed to exist; later versions are retitled 'Forget Forgetting...' and it is an ICLR 2026 poster. Sets up the unlimited-buffer, compute-bound regime and shows naive replay is strong. It also finds that abundant replay reduces plasticity, which bears on the plan's 'too much replay' side. This is the framing and baseline source.
- [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement (Jin & Ren)](https://arxiv.org/abs/2402.01865) — ICML 2024 (arXiv 2402.01865). Confirmed ID. Forecasts which upstream examples an update will make the model forget, from logit changes or representation inner products, and replays them. Its follow-up 2406.14026 shows forgetting matrices are low-rank. Prior art for the forgetting-proxy race and for forecast-guided replay.

**Queries run:** FOREVER arXiv 2601.03938 replay continual learning; MSSR arXiv 2603.09892 replay; adaptive replay ratio continual learning closed-loop forgetting signal how much to replay 2025; [extended] scaling law replay fraction number of tasks continual learning minimal replay forgetting pretrained; Klasson "Learn the Time to Learn" replay scheduling continual learning; Jin Ren "What Will My Model Forget" forecasting forgotten examples arXiv; "Scalable Strategies for Continual Learning with Replay" 2505.12512; Cho "Memory Is Not the Bottleneck" weight space consolidation continual learning 2502.07274; predicting forgetting continual learning pretrained ViT KL divergence proxy parameter distance per-class forgetting replay allocation 2025 2026; scaling laws forgetting fine-tuning pretraining data injection fraction Bethune 2025; [extended] compute-efficient replay class-incremental pretrained ViT LoRA how often to replay replay frequency forgetting 2025 2026 arXiv; continual learning replay triggered by drift detection controller adjusts replay ratio stored logits sentinel samples monitoring forgetting online

---

## 17. How budgeted 3D episodic memory forgets: a controlled study of grounded vs. LLM-rewrite consolidation across multi-scene deployments

*Added in the second round for an angle the first round missed.*

**Judge:** overall 4/10, tier C. Scores out of 5: novelty 2, low compute 4, impact 2, speed of first signal 3.
Its controlled replay of cached perception is a good methodological idea, but the space filled up between June and September 2026 (LT-Mem, eMEM, EMem, FAST-EQA, MemTree3D, Sequential EQA). It is also tied to scene graphs and EQA rather than the CUT3R, TTT3R and Point3R line the user cares about.

**Biggest weakness:** OpenEQA is known to be partly answerable without memory, so the headroom may be too small. Concatenated scenes are artificial, and a CLIP-feature coreset may match geometric coverage (kill criterion 1). Undetected budget sweeps in the full texts are likely.

**One line.** An RGB-D perception stream is cached once and then replayed through many memory-management policies at the same fixed storage budget. The study measures budget-accuracy and accuracy-vs-age (forgetting) curves for embodied QA as a deployment grows from 1 to 16+ scenes. It tests three things: whether geometry-grounded merge plus coverage eviction degrades gracefully where LLM-rewrite consolidation drifts, whether a simple set-cover model predicts where accuracy collapses, and whether a query-free recall proxy can stand in for costly VLM evaluation. Everything is inference-only, with frozen models of 1.5B parameters or less and an API VLM.

**Memory store.** The store studied is an external one: an episodic 3D scene memory made of image snapshots plus object nodes with 3D boxes and CLIP features. Inside this one store, consolidation runs across timescales, from raw snapshots (fast, expensive) to merged object nodes with a text gist (slow, cheap). That is the doc's open problem, consolidation across stores and timescales, posed with a fixed budget and evaluated by forgetting curves. The link to streaming 3D is an extension: rebuild the memory from CUT3R and TTT3R predicted poses and depth instead of ground truth, and measure how pose drift turns into duplicate nodes and wasted budget.

**Task.** The task is embodied question answering about episodes the agent has seen, under a fixed memory budget, over long multi-scene deployments.
(1) OpenEQA EM-EQA at K=1.
(2) OpenEQA-Stream, built here. K concatenated EM-EQA episode histories pass through one memory with a fixed global budget, and all questions are asked at the end. It runs in two conditions: scene-tagged and untagged. The untagged condition is the main one, because retrieval interference only shows up there.
(3) A small, secondary within-house revisit check on 3RScan rescans, testing whether budgeted eviction interacts with staleness. This is no longer a headline contribution, because LT-Mem, EMem and MemoryGuard cover staleness.

**Gap in the big picture.** The adversarial search refuted several parts of the original plan.

Revisit and staleness memory is largely scooped:
- LT-Mem (IROS 2026 best-paper finalist, https://arxiv.org/pdf/2608.19059) keeps cross-session object identity and chooses overwrite, hold or multi-hypothesis per object according to its volatility. Its Live/Delta/Meta memory comes with the LT-VQA temporal QA benchmark, and it reports an order of magnitude fewer tokens than baselines.
- EMem / EmbodiedMemory-Bench (https://arxiv.org/pdf/2609.28236) replaces outdated object relations in an entity graph.
- MemoryGuard (https://github.com/Jxy-yxJ/memoryguard) checks for stale locations under a bounded budget.
- "When Memory Lies" (https://arxiv.org/pdf/2608.04574) shows that stale memory can be worse than no memory.

Tiered consolidation into gists with a retention curve is in eMEM (https://arxiv.org/pdf/2606.03374). Bounded EQA memory is in FAST-EQA (https://arxiv.org/html/2602.15813), with k snapshots per target. Coverage or coreset selection under a fixed budget is in CoRDS (https://www.alphaxiv.org/abs/2605.14310), for streaming-video KV caches. Query-time key-frame selection over a 3D memory tree on OpenEQA is in MemTree3D (https://arxiv.org/pdf/2608.18009). Carrying memory across questions is in Sequential EQA (https://arxiv.org/pdf/2607.21571), which finds that spatially grounded memory is needed.

What remains open, as far as 10 searches found, is narrower. No work reports a controlled measurement of 3D episodic memory with:
(a) perception held fixed, so that only the management policy varies;
(b) the same global storage budget for every policy;
(c) multi-scene deployments, measured by budget-accuracy curves and accuracy-vs-age (backward-transfer) curves;
(d) a head-to-head of geometry-grounded merge/evict against LLM-rewrite text consolidation, the failure mode the user's doc cites (https://arxiv.org/pdf/2605.12978) but which nobody has tested in embodied 3D;
(e) an analytic predictor of the collapse point, and a validated query-free proxy for memory quality.

LT-Mem, eMEM and EMem each propose a system. None sweeps the budget or reports forgetting as a function of deployment length.

**Hypothesis.** Every policy gets the same budget: S_total snapshots plus a fixed allowance of text tokens.

H1 (curves differ in shape, not just level). On untagged OpenEQA-Stream, grounded consolidation (GC-Mem) has a larger area under the budget-accuracy curve (S from 4 to 64) than the best frame heuristic (equal-allocation uniform) and than a CoRDS-style feature coreset. The margin should be largest at small S.

H2 (predictable collapse). Accuracy as a function of K has a knee near K* ≈ S_total / m̄, where m̄ is the mean per-scene weighted set-cover size measured from the cache. Frame heuristics reach their knee earlier, at roughly S_total / (frames needed per scene).

H3 (drift). LLM-rewrite consolidation at equal token budget is non-monotonic in K and falls below its own K=1 score by K=16. Its forgetting is concentrated on old scenes and on spatial and attribute questions. GC-Mem does not fall below its own K=1 score.

H4 (proxy). The query-free recall of ground-truth object instances retained in memory correlates with LLM-Match across policies and budgets (Spearman ρ ≥ 0.7). That would let policies be designed without VLM calls.

Each hypothesis can fail on its own.

**Method.**

Perception (frozen, cached once). The 3D-Mem / ConceptGraphs-style stack runs on every 5th frame: YOLO-World, then SAM masks, then CLIP ViT-H features. Masks are back-projected with depth and pose. Every per-frame 3D object observation goes to disk. All management policies then run on the CPU over the identical replayed stream, so differences come only from management.

GC-Mem is kept deliberately simple: it is the grounded reference point, not a novelty claim. It runs online and never sees the questions.
(a) Merge: associate observations to object nodes by 3D IoU > τ and CLIP cosine > κ. Each node carries a fast evidence variable and a slow persistence count.
(b) Evict: when snapshots exceed S_total, run greedy budgeted weighted max-coverage over snapshots, which carries the (1-1/e) guarantee. Object weights are rarity × persistence.
(c) Residue: an evicted snapshot leaves text residue (labels, 3D centroid, room) charged against the token budget. Old scenes drift toward residue-only.
(d) Retrieve: 3D-Mem prefiltering picks C ≤ 8 snapshots plus residue for the API VLM. This runs in tagged and untagged conditions.

Comparators, all at equal budget and on the same cache:
- unbounded memory
- uniform subsampling, global and equal-allocation
- FIFO
- random
- 3D-Mem prefilter truncation
- surprise-gated write (Worth Remembering-style)
- FAST-EQA-style k snapshots per detected target
- CoRDS-style coreset on CLIP frame features, which is coverage without geometry and isolates what 3D grounding adds
- LT-Mem-style volatility-aware overwrite/hold, re-implemented in a simplified deterministic form
- eMEM-style tiered gist archival
- LLM-rewrite consolidation, in which captions are rewritten periodically to fit the token budget
- ReMEmbR-style caption memory with query-time retrieval

Analysis:
- budget-accuracy curves
- accuracy-vs-age curves and backward-transfer
- forgetting broken down by OpenEQA question category
- observed knee vs. predicted K*, from a toy set-cover model fitted on the cache alone
- correlation of the query-free proxy (GT-instance recall, duplicate-node rate) with LLM-Match

Secondary 3RScan revisit check: does coverage eviction preferentially remove stale views, and does that help or hurt current-state accuracy?

3D extension: replace ground-truth pose and depth with CUT3R and TTT3R, and measure the duplicate-node rate, the budget wasted on duplicates, and the resulting shift of the knee.

**Datasets.**

- OpenEQA EM-EQA (Majumdar et al., CVPR 2024): ScanNet and HM3D episode histories with human-written questions - https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html
- OpenEQA-Stream (built here): K in {1,4,16,32} concatenated EM-EQA histories, 3 scene orders per K, in tagged and untagged conditions
- 3RScan / RIO rescans (Wald et al., ICCV 2019), used only for a small templated revisit check - https://arxiv.org/pdf/1908.06109
- Optional: LT-VQA from LT-Mem, if publicly released (availability unverified), as an external revisit benchmark - https://arxiv.org/pdf/2608.19059

**Models / checkpoints.**

- Answering VLM, API only: gpt-4o-mini for sweeps and GPT-4o for the final runs, versions pinned and responses cached; a Gemini Flash-class model as a robustness check
- LLM-Match judge: the official OpenEQA judge prompt, held fixed
- Frozen perception, each at most 1.5B parameters: YOLO-World and SAM weights pinned in the 3D-Mem repo; CLIP ViT-H-14 (open_clip laion2b_s32b_b79k)
- Extension: the public CUT3R checkpoint and training-free TTT3R for predicted pose and depth

**Baselines.**

- Unbounded memory (upper bound); global and equal-allocation uniform subsampling; FIFO; random
- 3D-Mem snapshot memory with prefilter truncation - https://arxiv.org/html/2411.17735v2
- FAST-EQA-style bounded k-snapshots-per-target memory - https://arxiv.org/html/2602.15813
- CoRDS-style coreset selection under a fixed budget, applied to CLIP frame features - https://www.alphaxiv.org/abs/2605.14310
- LT-Mem-style volatility-aware overwrite/hold policy (simplified re-implementation) - https://arxiv.org/pdf/2608.19059
- eMEM-style tiered gist archival - https://arxiv.org/pdf/2606.03374
- Surprise-gated write at equal budget (Worth Remembering) - https://arxiv.org/pdf/2606.03787
- MemTree3D query-time key-frame selection, run over the budgeted memory - https://arxiv.org/pdf/2608.18009
- LLM-rewrite text consolidation (Mem0/ReMEmbR-style) at equal token budget
- OpenEQA published multi-frame VLM and Socratic baselines, for absolute reference

**Metrics.**

- LLM-Match (official judge) with paired-bootstrap 95% CIs and a judge-rerun noise floor
- Area under the budget-accuracy curve for S in {4,8,16,32,64}; retention ratio vs. unbounded memory
- Accuracy-vs-age curve and backward transfer (accuracy drop on scene-1 questions as K grows); knee location vs. predicted K* = S_total / m̄
- Per-category forgetting (OpenEQA categories: spatial, attribute, object recognition, and so on)
- Query-free proxy: GT-instance recall in memory and duplicate-node rate, with Spearman correlation to LLM-Match
- Cost: VLM input tokens per question, memory bytes, CPU write latency per frame; for the extension, ATE of CUT3R vs. TTT3R alongside the duplicate rate

**First experiment (go/no-go).**

This is a 10-day go/no-go on the ScanNet half of EM-EQA, with one GPU plus API access.

Days 1-3: cache the perception output, which takes about 3 GPU-hours. Compute m̄ and the predicted K* for S in {8, 32}.

Day 4: headroom check at K=1, comparing unbounded memory with 8 uniform frames using gpt-4o-mini.

Days 4-8: build OpenEQA-Stream with K in {1, 4, 16} and 3 orders each, in the untagged condition, at S_total in {8, 32}. Run 6 policies on the CPU: equal-allocation uniform, FIFO, CoRDS-style CLIP coreset, 3D-Mem truncation, LLM-rewrite, and GC-Mem. Use about 400 questions per configuration, which comes to roughly USD 30-100. Rerun the judge on 10% of answers.

Days 9-10: plot the budget-accuracy and age curves, compare the observed knee with K*, and compute the proxy correlation.

The go decision needs both of the following at K=16 and S=32 (untagged):
(i) GC-Mem beats both equal-allocation uniform and the CLIP coreset by at least 4 points, with the CI excluding 0, and the gap at K=16 is larger than at K=1;
(ii) either LLM-rewrite shows the predicted non-monotonic drift, or the observed knee lands within a factor of 2 of K*.
Without at least one of the two results in (ii), the paper is just another method with nothing to explain why it works.

**Kill criterion.**

Kill under either condition:
(1) Equal-allocation uniform or the CLIP-feature coreset comes within 2 LLM-Match points of GC-Mem at every (K, S) in the untagged condition. That would mean 3D grounding adds nothing over feature-space coverage, which CoRDS already covers.
(2) LLM-rewrite consolidation shows no drift, and the K* predictor misses the knee by more than a factor of 2. That would leave no analytic or empirical finding beyond method numbers already reported by LT-Mem, eMEM and FAST-EQA.

Do not pivot back to a staleness benchmark on 3RScan. That space is occupied by LT-Mem/LT-VQA, EMem and MemoryGuard.

**Compute.** 60 GPU-hours over 7 weeks.

**Risks.**

1. Contribution framing. The delta is now an evaluation and analysis contribution: a controlled budget/forgetting study, a K* predictor and a validated proxy. It is not a new memory system. Reviewers may see concatenated scenes as artificial. The untagged condition and the 3RScan within-house check partly answer this, but the realism critique stands.

2. Headroom. OpenEQA questions may be answerable from a few frames, and VLM reasoning, not memory, may be the bottleneck. The K>1 streams and small budgets create memory pressure, and the day-4 check catches the problem early.

3. Feature-coreset parity. A CLIP coreset may match the geometric policy (kill criterion 1).

4. Fidelity of re-implementations. Simplified versions of LT-Mem, eMEM and FAST-EQA may be called strawmen. Use the released code where it exists, and state the simplifications.

5. Judge noise of about 2 points. Mitigated with paired bootstrap and reruns.

6. Undetected overlap. A 2026 paper may already sweep budgets for 3D EQA memory. Only 10 searches were run, and full texts of LT-Mem, Sequential EQA and FAST-EQA were not read. Read them in week 1.

7. Pose drift in the CUT3R/TTT3R extension may break 3D association entirely. That would still be a reportable result.

8. API drift. Pin model versions and cache all responses.

GPU budget, about 60 hours: perception caching about 10 h, CUT3R/TTT3R about 10 h, debugging and slack about 40 h. The active-EQA stretch is dropped. API spend is about USD 300-600.

**Prior-work verdict: partially_novel; recommendation: pursue_with_pivot.** This is weaker than the original framing. The 2026 literature is crowded: LT-Mem (IROS 2026 finalist), eMEM, EMem-Bench, FAST-EQA, MemTree3D and Sequential EQA all appeared between June and September 2026. The novelty now rests on an analysis axis (budget sweep, forgetting vs. deployment length, the K* predictor, the proxy) and on the grounded-vs-LLM-rewrite comparison, not on any memory operation. I read only snippets, so the full texts of LT-Mem, FAST-EQA and Sequential EQA may contain budget ablations that narrow this further. The day-10 go criterion requires an analytic or drift finding, not just a method win, because a plain method win would not be publishable against these papers.

**What is taken and what is still new.** The original plan's method components are each covered by prior work:
- staleness invalidation and persistence variables: LT-Mem, EMem, MemoryGuard
- tiered gist consolidation: eMEM
- coverage eviction under a budget: CoRDS, in feature space
- bounded snapshot memory: FAST-EQA

The 3RScan Revisit-QA contribution is largely pre-empted by LT-Mem/LT-VQA and ChangingGrounding/Hypo3D on 3RScan.

What remains defensible is a controlled measurement study. Perception is cached and replayed, so policies differ only in management, all at an identical global budget. The study produces:
- budget-accuracy and accuracy-vs-age (backward-transfer) curves for 3D episodic memory over multi-scene untagged deployments
- the first embodied-3D test of LLM-rewrite consolidation drift against geometry-grounded merging
- a set-cover model that predicts the collapse point K* from the cache alone
- a validated query-free proxy for designing policies

Feature-space coverage (CoRDS-style) and LT-Mem- and eMEM-style policies serve as baselines rather than being ignored.

**Closest work:**

- [LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding](https://arxiv.org/pdf/2608.19059) — arXiv 2608.19059 / IROS 2026. Multi-session object-level memory with overwrite/hold/multi-hypothesis actions per object and Live/Delta/Meta tiers, plus the LT-VQA temporal QA benchmark; reports an order of magnitude fewer tokens. This scoops the original revisit/staleness hypothesis (H3). It does not sweep a fixed storage budget or measure forgetting vs. deployment length.
- [eMEM: A Hybrid Spatio-Temporal Memory System For Embodied Agents](https://arxiv.org/pdf/2606.03374) — arXiv 2606.03374 (2026). Tiered, hippocampus-inspired consolidation of raw observations into gists, with a retention-curve probe on ProcTHOR. This overlaps the residue/gist consolidation step. It reports no budget-accuracy curves and no head-to-head of grounded vs. LLM-rewrite consolidation.
- [EmbodiedMemory-Bench / EMem](https://arxiv.org/pdf/2609.28236) — arXiv 2609.28236 (2026). Long-horizon embodied memory benchmark whose entity graph replaces outdated object relations. Covers staleness tracking; no storage-budget study.
- [FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy](https://arxiv.org/html/2602.15813) — WACV 2026. Bounded EQA scene memory with a fixed k snapshots per target, plus a small k=3 vs. k=5 ablation. Single episode, no multi-scene forgetting analysis.
- [CoRDS: Coreset-based Representative and Diverse Selection for Streaming Video Understanding](https://www.alphaxiv.org/abs/2605.14310) — arXiv 2605.14310 (2026). Coverage/coreset selection under a fixed memory budget, for streaming-video KV caches. This pre-empts the coverage-eviction idea in feature space; the 3D-grounded version must beat it.
- [Beyond Episodic Evaluation: Memory Architectural Bottlenecks in Sequential Embodied Question Answering](https://arxiv.org/pdf/2607.21571) — arXiv 2607.21571 / IROS 2026. Carries memory across sequential questions and finds that spatially grounded 3D memory is needed. Single scene, no budget sweep.
- [Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering (MemTree3D)](https://arxiv.org/pdf/2608.18009) — arXiv 2608.18009 / ECCV 2026 (likely). Online 3D memory tree on OpenEQA with query-time key-frame selection. Query-time retrieval rather than budgeted write/evict.
- [MemoryGuard: bounded active memory maintenance for long-horizon embodied agents](https://github.com/Jxy-yxJ/memoryguard) — GitHub repo (2026). Bounded-budget embodied memory that checks for stale object locations in AI2-THOR. Overlaps the staleness/invalidate op.
- [When Memory Lies (spatial memory staleness in VLM agents)](https://arxiv.org/pdf/2608.04574) — arXiv 2608.04574 (2026). Shows that trusting raw memory can be worse than having no memory, on a FrozenLake testbed. Related to the drift/staleness hypothesis, but not 3D.
- [GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering](https://arxiv.org/pdf/2506.01174) — arXiv 2506.01174 (2025). Argues that compression fixed before the question is known loses what is needed, and edits the scene graph at inference time. A complementary query-time approach.

**Queries run:** memory budget embodied question answering 3D scene memory eviction long-term multi-scene 2026; bounded memory lifelong robot scene graph forgetting consolidation object persistence revisit changing environment question answering; 3RScan changing environment question answering benchmark object moved staleness embodied memory; LT-Mem volatility-aware spatio-temporal memory LT-VQA overwrite hold multi-hypothesis tokens baselines; "Beyond Episodic Evaluation" sequential embodied question answering memory architectural bottlenecks; fixed memory budget snapshot selection coverage 3D-Mem OpenEQA memory size accuracy tradeoff streaming video embodied; MemTree3D ECCV 2026 hierarchical memory tree 3D OpenEQA frame selection; embodied agent long-horizon memory across many environments fixed storage budget forgetting curve object-centric memory merge evict benchmark 2026 (extended); eMEM hybrid spatio-temporal memory embodied agents tiered consolidation eMEM-Bench retention probe; MemoryGuard bounded active memory maintenance embodied agents stale object location verification

---
