"""Check readout error P(-1)@x=0 for the heralded non-unitary BB1(GCR) as a function of
correction strength s, including s>1 (overcorrection). Expect a minimum at s=1 (exact
cancellation) and worse (larger) error for s>1. Also report subspace herald success."""
import numpy as np
from qutip import *
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=120
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_s(th,phi,s): return tensor(s*(th*ep/sq)*pOp,sigma(phi)).expm(method='dense')  # e^{+s c p sigma_phi}
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
psi0=tensor((displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit(),g)
def M(s):
    U=None
    for th,phi in BB1:
        op=Opx(th,phi)*Opy_s(th,phi,s)
        U=op if U is None else op*U
    return U
print(f"{'s':>5} {'P(-1)@0':>12} {'succ_sub(n<=40)':>16}")
for s in [0.5,0.75,1.0,1.10,1.25,1.5,2.0]:
    Ms=M(s); Mf=Ms.full(); nrm2=float((Ms*psi0).norm()**2)
    lam_s=np.linalg.svd(Mf[:, :80],compute_uv=False)[0]
    e=float(1-np.real(expect(ket2dm(py),(Ms*psi0).unit().ptrace(1))))
    print(f"{s:>5} {e:>12.3e} {nrm2/lam_s**2:>16.3e}")
