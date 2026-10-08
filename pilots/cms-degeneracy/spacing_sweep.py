import numpy as np, itertools
def run(gap, mode="shared", d=128, beta=0.5, eta=0.02, C=16, n_rep=4, M=32, R=64,
        Af=1.0, gamma=0.0, sleep=0, seed=0):
    rng=np.random.default_rng(seed)
    def unit(n):
        x=rng.standard_normal((n,d)); return x/np.linalg.norm(x,axis=1,keepdims=True)
    tk,tv=unit(M),unit(M)
    span=(n_rep-1)*gap+M+R+300; T=span
    dk,dv=unit(T),unit(T)
    ev=[(t,dk[t],dv[t]) for t in range(T)]
    for j in range(M):
        for r in range(n_rep):
            ev.append((span-R-j-(n_rep-1-r)*gap+0.5,tk[j],tv[j]))
    ev.sort(key=lambda e:e[0])
    Wf=np.zeros((d,d));Ws=np.zeros((d,d));G=np.zeros((d,d))
    for i,(t,k,v) in enumerate(ev):
        if mode=="slow_only":
            G+=np.outer(v-Ws@k,k)
        else:
            res=v-(Wf+Ws)@k; G+=np.outer(res,k); Wf+=beta*np.outer(res,k)
            if Af<1: Wf*=Af
        if (i+1)%C==0: Ws+=eta*G; G[:]=0
        if sleep and (i+1)%sleep==0: Wf*=(1-gamma)
    Ws+=eta*G
    return np.mean([tv[j]@Ws@tk[j] for j in range(M)])  # slow-level signal coefficient
gaps=[1,8,32,128,512,2048]
cfgs={"slow_only":dict(mode="slow_only"),
      "shared(HOPE-like)":dict(),
      "shared+fast decay Af=.99":dict(Af=0.99),
      "shared+fast decay Af=.97":dict(Af=0.97),
      "shared+sleep g=.5/64":dict(gamma=0.5,sleep=64),
      "shared+sleep g=1/256":dict(gamma=1.0,sleep=256)}
print("cfg".ljust(28)," ".join(f"{g:>7d}" for g in gaps))
for name,kw in cfgs.items():
    row=[np.mean([run(g,seed=s,**kw) for s in range(2)]) for g in gaps]
    print(name.ljust(28)," ".join(f"{x:7.4f}" for x in row),flush=True)
