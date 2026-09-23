"""User's refined proposal: split the 2pi correction into two pi pulses, so the
correction block is four pi pulses [pi@phi1, pi@3phi1, pi@3phi1, pi@phi1] (each
an identity at every readout peak x=2m|alpha|), herald the non-unitary ideal
i*sigma_phi correction on all four, and leave the target pi/2 pulse deterministic
(NOT heralded). Compare final-pulse = GCR (variant A) vs bare (variant B), and
against the unsplit [pi,2pi,pi] heralding and bare/ideal references."""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_sy(th):  return tensor(-1j*(th*ep/sq)*pOp,SY).expm(method='dense')                # unitary GCR target (sigma_y)
def Opy_nonU(th,phi): return tensor((th*ep/sq)*pOp,sigma(phi)).expm(method='dense')        # non-unitary e^{+c p sigma_phi}

# correction blocks
CORR_unsplit=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]           # original [pi,2pi,pi]
CORR_split  =[(np.pi,phi1),(np.pi,3*phi1),(np.pi,3*phi1),(np.pi,phi1)]  # 2pi -> pi+pi
TARGET=(np.pi/2,0.0)

def composite(corr, target, nonU_corr=True):
    M=None
    for (th,phi) in corr:
        op=Opx(th,phi)*(Opy_nonU(th,phi) if nonU_corr else qeye([Ncav,2]))
        M=op if M is None else op*M
    th,phi=target
    if target_mode=='gcr':  op=Opx(th,phi)*Opy_sy(th)
    elif target_mode=='bare': op=Opx(th,phi)
    M=op*M
    return M

N=121; alph=np.linspace(-2*sq,2*sq,N); m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
psi0=tensor(inits[int(np.argmin(np.abs(m)))],g)
def sweep(M):
    return np.array([np.real(expect(ket2dm(py),(M*tensor(it,g)).unit().ptrace(1))) for it in inits])
def succ(M):
    Mf=M.full(); nrm2=float((M*psi0).norm()**2)
    lg=np.linalg.svd(Mf,compute_uv=False)[0]; ls=np.linalg.svd(Mf[:,:80],compute_uv=False)[0]
    return nrm2/lg**2, nrm2/ls**2
i0=int(np.argmin(np.abs(m)))
res={}
for tag,corr,tmode in [("split-A",CORR_split,'gcr'),("split-B",CORR_split,'bare'),
                       ("unsplit-A",CORR_unsplit,'gcr')]:
    global target_mode; target_mode=tmode
    M=composite(corr,TARGET); P=sweep(M); res[tag]=P
    sg,ss=succ(M)
    print(f"[{time.time()-t0:.0f}s] {tag:10s} P(-1)@0={1-P[i0]:.3e}  succ_global={sg:.3e}  succ_sub(n<=40)={ss:.3e}")
np.savez("Paper_Data/readout_herald2.npz",m=m,**res)
print(f"[{time.time()-t0:.0f}s] done")
