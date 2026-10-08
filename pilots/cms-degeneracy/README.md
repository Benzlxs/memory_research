# CPU pilot for idea #5: update frequency is not retention

These are linear simulations of a two-level, continuum-memory-style (CMS) associative memory.
- The **fast** level is a delta rule that updates every event.
- The **slow** level accumulates the same shared residual and applies it every C events.
- Predictions use the sum of the two levels.
- At the session reset, the fast level is wiped and target items are read from the slow level alone.

Each script needs only NumPy and runs on CPU in seconds to a few minutes. The results below are from a run on 2026-10-08.

## 1. Degeneracy lemma (`lemma_check.py`)

With one shared loss, plain SGD and no per-level decay, the slow level is exactly a scaled copy of the fast level at every chunk boundary: W_s = (eta/beta) * W_f.

```
max rel ||Ws-(eta/beta)Wf|| at chunk boundaries: 3.3e-15
```

## 2. Retention after a reset vs spacing (`reset_retention.py`)

Recall score is 1 − squared error. Each of 32 targets is presented 4 times, `gap` events apart.

| gap | shared CMS, after reset | shared CMS, before reset | slow-only, after reset | independent levels, after reset |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.047 | 0.625 | 0.146 | 0.146 |
| 16 | 0.047 | 0.661 | 0.144 | 0.144 |
| 128 | 0.037 | 0.533 | 0.140 | 0.140 |
| 512 | 0.029 | 0.395 | 0.126 | 0.126 |
| 2048 | 0.027 | 0.361 | 0.089 | 0.089 |

Shared-loss CMS recalls well before the reset but keeps only about a third of the slow-only learner's recall after it. Spacing the repetitions does not help.

## 3. Breaking the symmetry (`spacing_sweep.py`)

This script measures the slow level's signal coefficient for the target items (mean of v_j · W_s k_j).

| config | gap 1 | 8 | 32 | 128 | 512 | 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slow only | 0.077 | 0.076 | 0.076 | 0.075 | 0.068 | 0.051 |
| shared (HOPE-like) | 0.024 | 0.024 | 0.023 | 0.019 | 0.014 | 0.014 |
| shared + fast decay 0.99 | 0.033 | 0.042 | 0.048 | 0.055 | 0.053 | 0.042 |
| shared + fast decay 0.97 | 0.042 | 0.059 | 0.065 | **0.067** | 0.062 | 0.047 |
| shared + downscale 0.5 every 64 | 0.033 | 0.042 | 0.049 | 0.055 | 0.053 | 0.042 |
| shared + reset every 256 | 0.025 | 0.028 | 0.038 | 0.047 | 0.045 | 0.037 |

Making the fast level forget, by continuous decay or periodic downscaling, restores a spacing optimum. With decay 0.97 the coefficient peaks at 0.067 at gap 128, 2.8× plain shared CMS.

Next step: test whether the same holds in a nonlinear model, DeltaNet plus two slow MLP levels on MQAR. See plan 5 in [`../../ideas/full-plans.md`](../../ideas/full-plans.md).
