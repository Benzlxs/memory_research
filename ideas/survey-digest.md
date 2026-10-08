# Digest: Hu et al., "Memory in the Age of AI Agents: A Survey — Forms, Functions and Dynamics" (arXiv 2512.13564; v1 15 Dec 2025, v2 13 Jan 2026)

Sources and their reliability. arXiv and the PDF are blocked from this environment; this digest was assembled from:
- [FIG] the survey's two overview figures (read directly from the companion repo github.com/Shichun-Liu/Agent-Memory-Paper-List, assets/main.png and assets/concept.png);
- [LIST] the companion repo's README paper list, organized exactly by the survey's Function x Form grid (github.com/Shichun-Liu/Agent-Memory-Paper-List, commit 4b45128, Mar 2026);
- [ABS] the abstract; [TOC] table-of-contents fragments seen in search results;
- [SNIP] search-result snippets quoting the survey; [SEC] third-party summaries (less reliable).

## 1. Scope (concept figure) [FIG]
Agent memory overlaps with but differs from:
- LLM memory: attention KV management, long-context processing (Mamba, Memformer, MoA, NSA).
- RAG: modular, graph and agentic RAG.
- Context engineering: tool-integrated reasoning, tool selection, communication protocols (MCP, A2A).
Overlap examples:
- Agent memory ∩ LLM memory: KV compression/reuse (AutoCompressor, SnapKV), self-reflection, few-shot prompting.
- Agent memory ∩ RAG ∩ context engineering: memory graphs (Zep, AriGraph), agentic memory (A-Mem, G-Memory), working memory (HiAgent, ReSum).
Agent-memory-specific: self-evolving memory (Memento, H2R), multimodal memory (Ella, ViloMem, M3-Agent), latent memory (MemoryLLM, M+, MemGen), parametric memory (Retroformer, Early Experience), RL-enabled memory (MemAgent, RMM, MemSearcher, MEM1, Mem-alpha, Memory-R1).

## 2. Forms — what carries memory [ABS, FIG, SNIP]
- Token-level (explicit, discrete, editable). Graded by topology [SNIP]: flat (1D logs, chunks), planar (2D single-layer graphs or trees), hierarchical (3D multi-level pyramids or layered graphs). Examples [FIG]: vector databases, knowledge graphs, multimodal RAG, extracted insights, context condensation and branching.
- Parametric (implicit weights). Examples [FIG]: model and knowledge editing (ROME, MEMIT, AlphaEdit, SERAC, Memory Decoder, MLP Memory); internalizing experiences (AgentBank, Agent-FLAN, Agent Lumos, Agent-R1, Early Experience, ETO, Chain-of-Agent, RAGEN, StarPO, Steca).
- Latent (hidden states, KV, continuous vectors). The figure shows four sub-areas:
  - KV generation (Titans, LM2, MemoRAG, Co-Processor);
  - KV reuse/compression (AutoCompressor, DMS, SnapKV, PyramidKV, H2O, Gist, KVPress, ClusterAttn);
  - latent memory generation (SoftCoT, MemGen, VisMem, LatentSeek, CoMEM, continuous GUI memory);
  - latent repository (MemoryLLM, M+, SirLLM, Memory^3, MemoryVLA, TrackVLA++).
- [SEC] trade-off: token-level stores dominate in practice. Parametric memory is efficient but suffers catastrophic forgetting; latent memory is efficient but opaque.

## 3. Functions — why agents need memory [ABS, SNIP]
- Factual: a persistent declarative knowledge base, about the user and about the environment.
- Experiential: procedural/episodic knowledge from past trajectories. It ranges from case-based (raw trajectories) through strategy-based (distilled workflows/insights) to skill-based (executable code/APIs), plus hybrids (e.g. ExpeL). There is a growing emphasis on strategy-based memory that transfers across tasks.
- Working: a capacity-limited, dynamically controlled scratchpad for one task or session. The work focuses on context compression and management: condensation, folding/branching (AgentFold, ContextFolding, HiAgent), and RL-trained context managers (MEM1, MemAgent).
- The survey says short-term vs long-term memory "emerge from the temporal patterns with which formation, evolution and retrieval are engaged", not from separate architectural modules [SNIP]. The figure overlays short-term ≈ working and long-term ≈ factual + experiential.

## 4. Dynamics — the lifecycle operators F, E, R [SNIP]
- Formation F transforms raw experience into information-dense units with long-term utility. Its routes are semantic summarization, knowledge distillation, structured construction, latent representation and parametric internalization.
- Evolution E = Consolidation ∪ Updating ∪ Forgetting. Updating covers conflict resolution in external stores and model editing for parametric memory. Sub-criteria for forgetting (e.g. time/frequency/importance) could not be confirmed from snippets.
- Retrieval R covers timing/intent, query construction, retrieval strategy and post-retrieval processing. Some systems retrieve once at task start, others iteratively.
- F, E and R "need not be invoked at every time step".

## 5. Positions and frontiers (Section 7) [TOC, ABS, SNIP, SEC]
- 7.1 From memory retrieval to memory generation [TOC]: agents synthesizing memory on demand rather than looking it up.
- 7.2 Automated memory management [TOC]: from handcrafted heuristics to learned, policy-driven pipelines; self-organizing hierarchies.
- 7.3 Reinforcement learning meets agent memory [TOC]: learn what to store, when to recall and how to consolidate (Memory-R1, Mem-alpha, MEM1, MemAgent, MemSearcher, RMM).
- Multimodal memory [ABS]: heterogeneous sensory inputs for embodied agents.
- Shared memory for multi-agent systems [ABS, SNIP]: a visible, manageable common factual foundation; private plus shared memories (Intrinsic Memory Agents).
- Memory for world models [SEC]: memory to build high-fidelity internal simulations for model-based reasoning and planning.
- Trustworthy memory [ABS, SEC]: privacy, explainability, robustness against memory leakage and hallucination.
- 7.8 Human-cognitive connections [TOC, SNIP]: current designs mirror the Atkinson–Shiffrin multi-store model (limited context plus external store), and Tulving's episodic/semantic/procedural split (logs / world knowledge / code skills).

## 6. Where the survey's own paper list is dense or sparse [LIST, counted from README]
| Function \ Form | Token-level | Parametric | Latent |
| --- | ---: | ---: | ---: |
| Factual | 84 | 16 | 8 |
| Experiential | 46 | 6 | 1 |
| Working | 14 | 2 | 22 |
Sparse cells: experiential–latent (1: "Auto-scaling Continuous Memory for GUI Agent"), working–parametric (2: StreamingLLM attention sinks, Lightning Attention), experiential–parametric (6: AgentEvolver, Early Experience, Scaling Agents via Continual Pre-training, ToolGen, Retroformer, a short/episodic/semantic memory machine), factual–latent (8).

Listed works directly relevant to the user's shortlist:
- Factual–parametric:
  - AlphaEdit, WISE, ELDER (lifelong editing), MEND/"Fast Model Editing at Scale", "Editing Factual Knowledge in LMs";
  - "Self-Updatable LLMs by Integrating Context into Model Parameters";
  - MAC ("Online Adaptation of LMs with a Memory of Amortized Contexts");
  - Memory Decoder, MLP Memory, MemLoRA;
  - "Pretraining with hierarchical memories" (long-tail vs common knowledge);
  - ELLA (lifelong learning).
- Factual–latent: M+, Memory^3, R3Mem, Memoria, SDM activations.
- Working–latent:
  - KV management: Scissorhands (the "persistence of importance hypothesis" for KV compression), H2O, SnapKV, RazorAttention;
  - memory architectures: Titans, LM2, Memorizing Transformers, Focused Transformer, MemoRAG;
  - embodied/visual: MemoryVLA (robot manipulation), XMem (video object segmentation with an Atkinson–Shiffrin memory model), VisMem;
  - other: MemGen, MEM1, Gist tokens, ICAE.
- Working–parametric: StreamingLLM (attention sinks).
- Experiential–token:
  - insight/procedure memories: Dynamic Cheatsheet, ACE, ReasoningBank, Memento, AWM, ExpeL, Reflexion, Memp, H2R;
  - "From RAG to Memory: Non-Parametric Continual Learning for LLMs" (HippoRAG 2);
  - MemEvolve (meta-evolution of memory systems), MemRL.
- Embodied/spatial: Mem2Ego, Embodied VideoAgent, MemoryVLA, TrackVLA++, JARVIS-1, "Planning from Imagination", "Context as Memory" (scene-consistent video generation), WorldMM.
- No streaming 3D reconstruction / geometric foundation model (CUT3R, TTT3R, Spann3R, Point3R, StreamVGGT) appears anywhere in the list or figures. The closest entries are XMem, MemoryVLA, Mem2Ego and "Context as Memory".

## 7. Critiques of the survey (third-party study note, [SEC])
- The factual vs experiential boundary is blurry.
- It has a complexity bias: it may undervalue simple brute-force baselines.
- It glosses over keeping a parametric memory (weights) synchronized with a token-level memory (DB) as both change.
- It says little about latency and engineering cost.
