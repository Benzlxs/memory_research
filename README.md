# memory_research

This repo holds research notes on memory management for continual learning.

- [`ideas/memory-management-ideas.md`](ideas/memory-management-ideas.md): a shortlist of 17 low-GPU project ideas.
  - It gives a ranked table, where to start, four research arcs, and one card per idea.
  - Each card covers the go/no-go experiment, the kill criteria, and the closest prior work.
- [`ideas/full-plans.md`](ideas/full-plans.md): the full plan for each idea, covering datasets, checkpoints, baselines, metrics, compute and risks.
- [`ideas/data/`](ideas/data/): the raw structured output of the brainstorm. This includes all 48 generated ideas, the prior-work checks and the judge's rankings.
- [`pilots/cms-degeneracy/`](pilots/cms-degeneracy/): a CPU pilot for idea #5, showing that multi-frequency memory levels trained on one shared loss do not consolidate across a session reset.

The brainstorm builds on the big-picture doc [Memory methods for continual learning: the big picture](https://claude.ai/artifact/JnG7owgYpsPUUfPsUhjnCr).
