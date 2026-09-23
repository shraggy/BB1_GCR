"""(a) Subspace-tailored heralding success. Global bound uses lambda_max = ||M|| over the whole
(truncated) space -- dominated by the unpopulated high-p tail. Fair bound: lambda_sub =
largest singular value of M restricted to the low-energy INPUT subspace (first n_low Fock (x)
qubit), since a valid Kraus M/lambda <= I is only needed on states we actually feed in.
success(n_low) = ||M psi||^2 / lambda_sub(n_low)^2.  Do it for s=0.5,0.75,1.0."""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_nonU(th,phi,s): return tensor(-1j*s*(th*ep/sq)*pOp,1j*sigma(phi)).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def Mcomposite(s):
    U=None
    for th,phi in BB1:
        op=Opx(th,phi)*Opy_nonU(th,phi,s)
        U=op if U is None else op*U
    return U
psi0=tensor((displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit(),g)

# input-state containment vs Fock cutoff (justifies the subspace)
osc0=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
pn=np.abs(osc0.full().ravel())**2
print("input oscillator Fock containment: ",{nl:round(float(pn[:nl].sum()),5) for nl in [15,20,30,40,60]})

nls=[15,20,30,40,60,80,100]
print(f"\n{'s':>4} {'||Mpsi||^2':>10} {'n_low':>6} {'lambda_sub':>11} {'success':>10}")
res={}
for s in [0.5,0.75,1.0]:
    M=Mcomposite(s); Mf=M.full(); nrm2=float((M*psi0).norm()**2)
    for nl in nls:
        lam=np.linalg.svd(Mf[:, :2*nl], compute_uv=False)[0]  # restrict input cols to first nl Fock (x) 2 qubits
        succ=nrm2/lam**2
        res[(s,nl)]=(lam,succ)
        print(f"{s:>4} {nrm2:>10.2f} {nl:>6} {lam:>11.2f} {succ:>10.2e}")
    print()
# summary: fair success at n_low=40 (contains ~all of the input) vs global (n_low=100)
print("=== fair (n_low=40) vs global (n_low=100) success ===")
for s in [0.5,0.75,1.0]:
    print(f" s={s}: fair(n40)={res[(s,40)][1]:.2e}   global(n100)={res[(s,100)][1]:.2e}   gain x{res[(s,40)][1]/res[(s,100)][1]:.0f}")
np.savez("Paper_Data/subspace_herald.npz",nls=np.array(nls),
         **{f"succ_s{s}":np.array([res[(s,nl)][1] for nl in nls]) for s in [0.5,0.75,1.0]})
print(f"[{time.time()-t0:.0f}s] done")
