"""
Root-cause analysis of why PHYSICAL (unitary) GCR fails in the composite, + fixes for BB1.

Hypothesis:
  The exact GCR precorrection is exp(-i c p_hat (x) [1j sigma_phi]) = exp(c p_hat sigma_phi),
  a NON-unitary "imaginary conditional displacement" (an (x-alpha)-reweighting). The physical
  unitary gate can only do a REAL conditional displacement exp(-i c p_hat sigma_perp), which
  equals the ideal only to FIRST ORDER in the pulse angle. Consequences:
    (A) it diverges from the ideal as the pulse angle grows (worst at the 2pi pulse);
    (B) each physical pulse leaves the oscillator off the clean Gaussian the next pulse needs.

PART A  single pulse about sigma_x, vary angle theta: infidelity of bare / vec(physical) /
        vecvecf(ideal) to the exact rotation.  Expect vec ~ vecvecf at small theta, diverging
        at large theta; bare worst.  Isolates cause (A).

PART B  fixes for BB1 (readout, net pi/2), operating-point error:
   B1  split each BB1 pulse into K equal sub-pulses (physical vec GCR), K=1,2,4,8  -> does
       staying in the small-angle regime let physical GCR approach the ideal?
   B2  rescale the physical precorrection amplitude by s (K=1); scan s -> is the first-order
       amplitude even optimal for the large composite pulses?
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=120
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm(method='dense')
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
alpha0=sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,phi,axis,scale=1.0): return tensor(-1j*scale*(th*ep/sq)*pOp,axis).expm(method='dense')
clean=(displace(Ncav,alpha0/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()

# ---------- PART A : single pulse vs angle ----------
print(f"[{time.time()-t0:.0f}s] PART A single-pulse angle sweep")
thetas=np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0])*np.pi  # up to 4pi
def single(theta,mode):
    vecf=sigmaz(); vec=1j*sigma(0.0)*vecf; axis=(vec*vecf) if mode=='vecvecf' else vec
    ox=Opx(theta,0.0)
    op = ox if mode=='bare' else ox*Opy(theta,0.0,axis)
    s=(op*tensor(clean,g)).unit()
    ideal_q=(-1j*theta/2*sigmax()).expm()*g
    return 1-fidelity(s.ptrace(1),ket2dm(ideal_q))
A={m:np.array([single(th,m) for th in thetas]) for m in ['bare','vec','vecvecf']}
print("  theta/pi :", " ".join(f"{t/np.pi:.2f}" for t in thetas))
for m in ['bare','vec','vecvecf']:
    print(f"  {m:<8}:", " ".join(f"{v:.1e}" for v in A[m]))

# ---------- PART B : fixes for BB1 ----------
phi1=np.arccos(-1/8)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
N=31; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def build(seq,mode,scale=1.0):
    ops=[]; vecf=sigmaz()
    for th,phi in seq:
        ox=Opx(th,phi)
        if mode=='bare': ops.append(ox)
        else:
            vec=1j*sigma(phi)*vecf; axis=(vec*vecf) if mode=='vecvecf' else vec
            ops.append(ox*Opy(th,phi,axis,scale))
        R=rot_xy(2*th,phi); vecf=R*vecf*R.dag()
    return ops
def sweepP(ops): return np.array([ (lambda s: np.real(expect(ket2dm(py),s.ptrace(1))))(_run(ops,it)) for it in inits])
def _run(ops,init):
    s=tensor(init,g)
    for U in ops: s=(U*s).unit()
    return s
def op_err(P,ref):
    mask=np.abs(ref-np.round(ref))<0.07; tgt=np.round(ref)
    return np.mean(np.abs(P-tgt)[mask])

Pbare=sweepP(build(BB1,'bare')); Pideal=sweepP(build(BB1,'vecvecf'))
ebare=op_err(Pbare,Pideal); eideal=op_err(Pideal,Pideal)
print(f"\n[{time.time()-t0:.0f}s] PART B1 pulse-splitting (physical vec), op-err vs K:")
print(f"    reference: bare={ebare:.4f}  ideal vec*vecf={eideal:.4f}")
splitres={}
for K in [1,2,4,8]:
    seq=[]
    for th,phi in BB1: seq += [(th/K,phi)]*K
    P=sweepP(build(seq,'vec')); e=op_err(P,Pideal); splitres[K]=(P,e)
    print(f"    K={K:<2} ({len(seq)} sub-pulses): physical-GCR op-err={e:.4f}")

print(f"\n[{time.time()-t0:.0f}s] PART B2 precorrection amplitude scale s (K=1 physical vec):")
scales=[0.0,0.25,0.5,0.75,1.0,1.5,2.0]
scaleres={}
for s in scales:
    P=sweepP(build(BB1,'vec',scale=s)); e=op_err(P,Pideal); scaleres[s]=e
    print(f"    s={s:<4}: op-err={e:.4f}" + ("  (s=0 -> bare)" if s==0 else ("  (s=1 -> standard physical GCR)" if s==1 else "")))

np.savez("Paper_Data/rootcause.npz", thetas=thetas, **{f"A_{k}":v for k,v in A.items()},
         m=m, Pbare=Pbare, Pideal=Pideal,
         **{f"split_{K}":splitres[K][0] for K in splitres}, scales=np.array(scales),
         scale_err=np.array([scaleres[s] for s in scales]))

# ---------- plots ----------
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
# A
for m_ in ['bare','vec','vecvecf']:
    ax[0].plot(thetas/np.pi,A[m_],'-o',ms=4,label=m_)
ax[0].set_yscale('log'); ax[0].set_xlabel(r"pulse angle $\theta/\pi$"); ax[0].set_ylabel("single-pulse infidelity")
ax[0].axvline(2,ls=':',color='gray'); ax[0].text(2.02,ax[0].get_ylim()[1]*0.3,'2π pulse',fontsize=8)
ax[0].set_title("(A) Root cause: physical vec\ntracks ideal only at small angle"); ax[0].legend(); ax[0].grid(alpha=0.3)
# B1
Ks=sorted(splitres); ax[1].plot(Ks,[splitres[K][1] for K in Ks],'-o',color='seagreen',label='physical GCR, split')
ax[1].axhline(ebare,ls='--',color='cornflowerblue',label=f'bare BB1 ({ebare:.3f})')
ax[1].axhline(eideal,ls='--',color='firebrick',label=f'ideal vec*vecf ({eideal:.3f})')
ax[1].set_xlabel("K (sub-pulses per BB1 pulse)"); ax[1].set_ylabel("operating-point error")
ax[1].set_title("(B1) Fix: split pulses to stay\nin small-angle regime"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
# B2
ax[2].plot(scales,[scaleres[s] for s in scales],'-o',color='purple')
ax[2].axhline(ebare,ls='--',color='cornflowerblue',label=f'bare ({ebare:.3f})')
ax[2].axhline(eideal,ls='--',color='firebrick',label=f'ideal ({eideal:.3f})')
ax[2].set_xlabel("precorrection amplitude scale s"); ax[2].set_ylabel("operating-point error")
ax[2].set_title("(B2) Fix: re-optimize physical\nprecorrection amplitude"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("Paper_Figures/rootcause.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/rootcause.png")
