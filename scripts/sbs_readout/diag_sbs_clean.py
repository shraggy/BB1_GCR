import numpy as np
from qutip import *
import sbs_lib as sb

Ncav = 100
L0 = sb.logical(0, Ncav); L1 = sb.logical(1, Ncav)

def Fenv(rho):
    F0 = float(np.real(expect(ket2dm(L0), rho)))
    F1 = float(np.real(expect(ket2dm(L1), rho)))
    return max(F0,F1), F0, F1

def run(start_label, cond, nrounds=10, order='xp'):
    rho = ket2dm(L0) if start_label==0 else ket2dm(L1)
    trace=[]
    fe,f0,f1 = Fenv(rho); trace.append((fe,f0,f1, sb.impurity(rho)))
    for n in range(nrounds):
        axis = order[n % len(order)]
        rho = sb.sbs_round(rho, axis, cond, Ncav)
        fe,f0,f1 = Fenv(rho)
        trace.append((fe,f0,f1, sb.impurity(rho)))
    return trace

for start in [0,1]:
    for cond in ['noiseless','complete']:
        tr = run(start, cond)
        print(f"=== start=L{start}  cond={cond} ===")
        for i,(fe,f0,f1,im) in enumerate(tr):
            print(f"  round {i:2d}: F_env={fe:.4f}  F(L0)={f0:.4f} F(L1)={f1:.4f}  impurity={im:.4e}")
