"""
Correct physical GCR precorrection axis (user's rule):
  sigma_gamma must be perpendicular to (a) the pulse rotation axis phi_hat, and
  (b) n_hat = the Bloch axis whose eigenstate the qubit is in at the START of the pulse,
  where n_hat is obtained by propagating the NO-ERROR preceding rotations on the qubit
  (start |g> = +z; trivial for the first pulse).
  => sigma_gamma_hat = normalize(n_hat x phi_hat)   (always a realizable Hermitian axis)

This replaces my earlier reverse-engineered 'vec' (1j sigma(phi) vecf, 2theta frame update),
which was correct only for BB1. Test whether the correct axis makes physical GCR UNITARY and
WORKING for both BB1 and SCROFULOUS. Also compare sign(n x phi vs phi x n).
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

def build(seq,mode,sign=+1):
    ops=[]; n=np.array([0,0,1.0])  # |g> Bloch vector
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare':
            ops.append(ox)
        else:
            if mode=='ideal':
                axis=1j*sigma(phi)                          # non-unitary exact
            else:  # 'phys' : correct Hermitian axis n x phi
                gamma=sign*np.cross(n,phiv); gn=np.linalg.norm(gamma)
                axis = svec(gamma/gn) if gn>1e-9 else 0*SX  # if n||phi, no correction
            ops.append(ox*Opy(th,axis) if not isinstance(axis,(int,float)) else ox)
        n=rot_bloch(n,phiv,th)   # no-error rotation by theta about phi
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
# sanity: print n_hat and sigma_gamma per pulse
print("=== n_hat (qubit Bloch axis at start of pulse) and sigma_gamma = n x phi ===")
for c,seq in SEQS.items():
    n=np.array([0,0,1.0]); print(f"  {c}:")
    for th,phi in seq:
        phiv=np.array([np.cos(phi),np.sin(phi),0.0]); gam=np.cross(n,phiv)
        print(f"    theta={th/np.pi:.2f}pi phi={phi/np.pi:.2f}pi | n_hat=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}) "
              f"| gamma=n x phi=({gam[0]:+.2f},{gam[1]:+.2f},{gam[2]:+.2f}) |gamma|={np.linalg.norm(gam):.2f}")
        n=rot_bloch(n,phiv,th)

N=31; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

print(f"\n[{time.time()-t0:.0f}s] scheme                     op-err   ||U^dU-I||   unit?")
res={}
for c,seq in SEQS.items():
    Pideal=sweepP(build(seq,'ideal'))
    def oe(P):
        mask=np.abs(Pideal-np.round(Pideal))<0.07; return np.mean(np.abs(P-np.round(Pideal))[mask])
    for mode,sign,lab in [('bare',1,'bare'),('phys',+1,'phys (n x phi)'),('phys',-1,'phys (phi x n)'),('ideal',1,'ideal (non-unitary)')]:
        ops=build(seq,mode,sign); P=sweepP(ops); u=(Ufull(ops).dag()*Ufull(ops)-I2).norm()
        res[(c,lab)]=(P,oe(P),u)
        print(f"  {c:<11}{lab:<20} {oe(P):.4f}   {u:.2e}   {'UNITARY' if u<1e-6 else 'NON-UNITARY'}")

np.savez("Paper_Data/correct_axis.npz", m=m, **{f"{c}_{lab}".replace(' ','_').replace('(','').replace(')','').replace('x','X'):res[(c,lab)][0] for (c,lab) in res})

fig,ax=plt.subplots(1,2,figsize=(15,5),sharey=True)
for i,c in enumerate(SEQS):
    for lab,(cl,ls) in {'bare':('cornflowerblue','-'),'phys (n x phi)':('seagreen','-'),
                        'phys (phi x n)':('orange','-.'),'ideal (non-unitary)':('firebrick','--')}.items():
        P,e,u=res[(c,lab)]
        ax[i].plot(m,P,ls,color=cl,lw=2.2,label=f"{lab} (op {e:.3f}, {'U' if u<1e-6 else 'NU'})")
    ax[i].set_title(f"{c} (net pi/2) — correct axis sigma_gamma=n x phi"); ax[i].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$")
    ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
ax[0].set_ylabel("P(+1)")
fig.tight_layout(); fig.savefig("Paper_Figures/correct_axis.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/correct_axis.png")
