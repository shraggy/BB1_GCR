"""
Definitive physical-GCR test: CORRECT axis (sigma_gamma = n_hat x phi_hat, n_hat propagated
through no-error rotations) + PULSE SPLITTING (small-angle regime), respecting determinism
(split correction pulses, keep the final target pulse whole).

For BB1 and SCROFULOUS (net pi/2): op-err and unitarity vs K (sub-pulses per correction pulse).
Question: does physical GCR now converge toward the ideal and beat bare? Does SCROFULOUS work?
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=120
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
def rot_bloch(n,axis,angle):
    k=np.array(axis,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(angle)+np.cross(k,n)*np.sin(angle)+k*np.dot(k,n)*(1-np.cos(angle))

def build(seq,mode):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:  # phys, correct axis
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            ops.append(ox*Opy(th,svec(gam/gn)) if gn>1e-9 else ox)
        n=rot_bloch(n,phiv,th)
    return ops
def Ufull(ops):
    U=ops[0]
    for o in ops[1:]: U=o*U
    return U

phi1=np.arccos(-1/8); d2r=np.pi/180
SEQS={
 'BB1':        [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)],
 'SCROFULOUS': [(115.2*d2r,62*d2r),(np.pi,280.6*d2r),(115.2*d2r,62*d2r),(np.pi/2,0.0)],
}
def corr_split(seq,K):  # split all but the LAST pulse (determinism: keep target whole)
    out=[]
    for i,(th,phi) in enumerate(seq):
        out += [(th,phi)] if i==len(seq)-1 else [(th/K,phi)]*K
    return out

N=31; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

Ks=[1,2,4,8]
results={}
for c,seq in SEQS.items():
    Pideal=sweepP(build(seq,'ideal'))
    def oe(P):
        mask=np.abs(Pideal-np.round(Pideal))<0.07; return np.mean(np.abs(P-np.round(Pideal))[mask])
    ebare=oe(sweepP(build(seq,'bare'))); eideal=oe(Pideal)
    print(f"\n{c}: bare={ebare:.4f}  ideal={eideal:.4f}")
    errs=[]
    for K in Ks:
        ops=build(corr_split(seq,K),'phys'); P=sweepP(ops)
        u=(Ufull(ops).dag()*Ufull(ops)-I2).norm(); e=oe(P); errs.append(e)
        print(f"   K={K}: correct-axis physical op-err={e:.4f}  ||U^dU-I||={u:.1e} {'UNITARY' if u<1e-6 else 'NONUNIT'}")
    results[c]=(ebare,eideal,errs)

fig,ax=plt.subplots(1,2,figsize=(13,5))
for i,c in enumerate(SEQS):
    ebare,eideal,errs=results[c]
    ax[i].plot(Ks,errs,'-o',color='seagreen',lw=2,label='physical GCR (correct axis, split)')
    ax[i].axhline(ebare,ls='--',color='cornflowerblue',label=f'bare ({ebare:.3f})')
    ax[i].axhline(eideal,ls='--',color='firebrick',label=f'ideal ({eideal:.3f})')
    ax[i].set_title(f"{c} (net pi/2): correct axis + splitting"); ax[i].set_xlabel("K (sub-pulses per correction pulse)")
    ax[i].set_ylabel("operating-point error"); ax[i].legend(fontsize=9); ax[i].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("Paper_Figures/correct_split.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/correct_split.png")
