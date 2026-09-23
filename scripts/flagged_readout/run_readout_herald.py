"""Test the user's readout proposal: herald ONLY the 3 BB1 correction pulses
(non-unitary ideal pre-correction i*sigma_phi), keep the final target pulse
deterministic. Two final-pulse variants:
  A 'gcr'  : final = deterministic GCR(pi/2) with unentangled-axis sigma_gamma = n x phi
  B 'bare' : final = bare position rotation only (no momentum pre-correction)
Compare against: bare BB1 (all unitary, no correction) and ideal (all 4 non-unitary).
Report: herald success prob (global lambda=||M|| and subspace n<=40) and the
success-branch readout response P(+1) vs <x>, plus operating-point P(-1)@x=0."""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_axis(th,ax): return tensor(-1j*(th*ep/sq)*pOp,svec(ax)).expm(method='dense')      # unitary
def Opy_nonU(th,phi): return tensor((th*ep/sq)*pOp,sigma(phi)).expm(method='dense')        # e^{+c p sigma_phi}, non-unitary
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]

def composite(mode):
    """mode: 'bare' (no corrections), 'ideal' (all4 nonU), 'A' (3 nonU + GCR final),
             'B' (3 nonU + bare final)."""
    M=None; n=np.array([0,0,1.0])
    for i,(th,phi) in enumerate(BB1):
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        gam=np.cross(n,phiv); gam=gam/np.linalg.norm(gam)
        if mode=='bare':                      op=ox
        elif mode=='ideal':                   op=ox*Opy_nonU(th,phi)
        elif mode in ('A','B'):
            if i<3:                           op=ox*Opy_nonU(th,phi)          # heralded correction
            else:                             op=ox*Opy_axis(th,gam) if mode=='A' else ox   # final pulse
        M=op if M is None else op*M
        n=rot_bloch(n,phiv,th)
    return M

# inputs across the readout range
N=121; alph=np.linspace(-2*sq,2*sq,N); m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
psi0=tensor(inits[int(np.argmin(np.abs(m)))],g)   # x=0 input

def sweep_success(M):
    out=[]
    for it in inits:
        s=(M*tensor(it,g)).unit()             # herald-success branch (renormalized)
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

def succ_probs(M):
    Mf=M.full(); nrm2=float((M*psi0).norm()**2)
    lam_g=np.linalg.svd(Mf,compute_uv=False)[0]
    lam_s=np.linalg.svd(Mf[:,:80],compute_uv=False)[0]   # n<=40 input subspace
    return nrm2/lam_g**2, nrm2/lam_s**2

res={}
for mode in ['bare','ideal','A','B']:
    M=composite(mode); P=sweep_success(M); res[mode]=P
    i0=int(np.argmin(np.abs(m))); pm1=1-P[i0]
    if mode in ('ideal','A','B'):
        sg,ss=succ_probs(M)
        print(f"[{time.time()-t0:.0f}s] {mode:5s}  P(-1)@0={pm1:.3e}  succ_global={sg:.3e}  succ_subspace(n<=40)={ss:.3e}")
    else:
        print(f"[{time.time()-t0:.0f}s] {mode:5s}  P(-1)@0={pm1:.3e}  (deterministic, success=1)")
np.savez("Paper_Data/readout_herald.npz", m=m, **{k:res[k] for k in res})
print(f"[{time.time()-t0:.0f}s] saved Paper_Data/readout_herald.npz")
