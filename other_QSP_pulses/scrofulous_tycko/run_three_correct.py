"""
REVISED three-composite figure using the CORRECT precorrection axis sigma_gamma = n_hat x phi_hat
(supersedes run_three.py / three_composites.png, whose SCROFULOUS 'vec' used the wrong axis).
BB1 / SCROFULOUS / TYCKO, each: bare, physical vec (correct axis), ideal vec*vecf. Net theta=pi/2.
Reports op-err and unitarity per curve so the SCROFULOUS correction can be verified.
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=140
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
        elif mode=='vecvecf': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:  # correct physical axis n x phi
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
 'TYCKO':      [(385*d2r,0.0),(320*d2r,np.pi),(25*d2r,0.0),(np.pi/2,0.0)],
}
MODES=[('bare','bare (no GCR)','cornflowerblue','-'),
       ('vec','physical vec (correct axis n x phi)','seagreen','-.'),
       ('vecvecf','ideal vec*vecf','firebrick','--')]

N=41; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

P={}; U={}
for c,seq in SEQS.items():
    for mode,_,_,_ in MODES:
        ops=build(seq,mode); P[(c,mode)]=sweepP(ops); U[(c,mode)]=(Ufull(ops).dag()*Ufull(ops)-I2).norm()
    print(f"[{time.time()-t0:.0f}s] {c} done")

print("\n=== REVISED (correct axis) op-err + unitarity ===")
summary={}
for c,seq in SEQS.items():
    pid=P[(c,'vecvecf')]; mask=np.abs(pid-np.round(pid))<0.07; tgt=np.round(pid)
    print(f"{c}:")
    for mode,lab,_,_ in MODES:
        e=np.mean(np.abs(P[(c,mode)]-tgt)[mask]); summary[(c,mode)]=e
        u=U[(c,mode)]
        print(f"   {lab:<38} op-err={e:.4f}  ||U^dU-I||={u:.1e}  {'UNITARY' if u<1e-6 else 'NON-UNIT'}")

np.savez("Paper_Data/three_correct.npz", m=m, **{f"{c}_{mode}":P[(c,mode)] for c in SEQS for mode,_,_,_ in MODES})

fig,ax=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for i,(c,seq) in enumerate(SEQS.items()):
    for mode,lab,col,ls in MODES:
        u=U[(c,mode)]; tag='U' if u<1e-6 else 'NU'
        ax[i].plot(m,P[(c,mode)],ls,color=col,lw=2.4 if mode=='vecvecf' else 2,
                   label=f"{lab.split('(')[0].strip()} (op {summary[(c,mode)]:.3f}, {tag})")
    ax[i].set_title(f"{c} (net $\\theta=\\pi/2$)"); ax[i].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$")
    ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7.5,loc='lower center')
ax[0].set_ylabel("P(+1)")
fig.suptitle("REVISED: correct axis $\\sigma_\\gamma=\\hat n\\times\\hat\\phi$  (bare / physical vec / ideal vec*vecf)",fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/three_composites_correct.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/three_composites_correct.png")
