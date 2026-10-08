import numpy as np
d=64;beta=0.5;eta=0.02;C=16;rng=np.random.default_rng(1)
Wf=np.zeros((d,d));Ws=np.zeros((d,d));G=np.zeros((d,d));errs=[]
for i in range(4000):
    k=rng.standard_normal(d);k/=np.linalg.norm(k);v=rng.standard_normal(d);v/=np.linalg.norm(v)
    res=v-(Wf+Ws)@k;G+=np.outer(res,k);Wf+=beta*np.outer(res,k)
    if (i+1)%C==0:
        Ws+=eta*G;G[:]=0
        errs.append(np.linalg.norm(Ws-(eta/beta)*Wf)/np.linalg.norm(Ws))
print("max rel ||Ws-(eta/beta)Wf|| at chunk boundaries:",max(errs))
