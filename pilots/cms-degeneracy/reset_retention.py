"""Quick CPU sanity check: does a 2-level shared-residual (CMS-style) linear memory
show residual starvation / a spacing effect at session reset, and does periodic
downscaling of the fast level help?

Fast level: delta rule every event, lr beta.  Slow level: accumulates the shared
residual gradient and applies it every C events with lr eta.  Prediction uses
W_f + W_s (parallel levels).  At the end of the session W_f is reset to 0 and we
read target items from W_s only.
"""
import numpy as np, sys

def run(gap, d=128, beta=0.5, eta=0.02, C=16, n_rep=4, M=32, R=64, load=1,
        mode="shared", gamma=0.0, sleep_every=0, seed=0, decay=0.0):
    rng = np.random.default_rng(seed)
    def unit(n):
        x = rng.standard_normal((n, d)); return x / np.linalg.norm(x, axis=1, keepdims=True)
    tk, tv = unit(M), unit(M)
    span = (n_rep - 1) * gap + M + R + 200
    T = int(span * load)
    events = []  # (time, key, value, target_id)
    dk, dv = unit(T), unit(T)
    for t in range(T):
        events.append((t / load, dk[t], dv[t], -1))
    end = span
    for j in range(M):
        for r in range(n_rep):
            tt = end - R - j - (n_rep - 1 - r) * gap + 0.5
            events.append((tt, tk[j], tv[j], j))
    events.sort(key=lambda e: e[0])
    Wf = np.zeros((d, d)); Ws = np.zeros((d, d)); G = np.zeros((d, d))
    for i, (t, k, v, j) in enumerate(events):
        pred = (Wf + Ws) @ k
        res = v - pred
        if mode == "shared":
            G += np.outer(res, k)
            Wf += beta * np.outer(res, k)
        elif mode == "fast_only":
            Wf += beta * np.outer(v - Wf @ k, k)
        elif mode == "slow_only":
            G += np.outer(v - Ws @ k, k)
        elif mode == "independent":  # each level learns the full target (no shared residual)
            Wf += beta * np.outer(v - Wf @ k, k)
            G += np.outer(v - Ws @ k, k)
        if decay > 0:
            Wf *= (1 - decay)
        if (i + 1) % C == 0:
            Ws += eta * G; G[:] = 0
        if sleep_every and (i + 1) % sleep_every == 0 and gamma > 0:
            Wf *= (1 - gamma)
    Ws += eta * G
    rec_post = np.mean([1 - np.sum((Ws @ tk[j] - tv[j])**2) for j in range(M)])
    rec_pre = np.mean([1 - np.sum(((Wf + Ws) @ tk[j] - tv[j])**2) for j in range(M)])
    if mode == "independent":
        rec_pre = rec_post
    return rec_post, rec_pre

if __name__ == "__main__":
    gaps = [1, 4, 16, 64, 128, 256, 512, 1024, 2048]
    print("gap | shared post | shared pre | slow_only post | indep post")
    for g in gaps:
        s = np.mean([run(g, seed=s, mode="shared") for s in range(3)], axis=0)
        so = np.mean([run(g, seed=s, mode="slow_only") for s in range(3)], axis=0)
        ind = np.mean([run(g, seed=s, mode="independent") for s in range(3)], axis=0)
        print(f"{g:5d} | {s[0]:.3f} | {s[1]:.3f} | {so[0]:.3f} | {ind[0]:.3f}", flush=True)
