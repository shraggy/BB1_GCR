"""
Verify how BB1(GCR) physical changes between the OLD (wrong) axis and the CORRECT axis.
  old  : vec = 1j sigma(phi) vecf, vecf updated by rot_xy(2 theta, phi)  [uses n_hat=+z always for BB1]
  new  : sigma_gamma = n_hat x phi_hat, n_hat propagated by the actual no-error rotation (theta)
Show op-err, unitarity, and the per-pulse precorrection axis for both, on the same grid.
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
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def rot_bloch(n,axis,angle):
    k=np.array(axis,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(angle)+np.cross(k,n)*np.sin(angle)+k*np.dot(k,n)*(1-np.cos(angle))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
phi1=np.arccos(-1/8)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]

def build(mode):
    ops=[]; vecf=SZ; n=np.array([0,0,1.0])
    print(f"  axes for mode={mode}:")
    for th,phi in BB1:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        elif mode=='old':
            vec=1j*sigma(phi)*vecf; ops.append(ox*Opy(th,vec))
            # report axis as bloch components
            print(f"    theta={th/np.pi:.2f}pi: old axis ~ 1j*sigma(phi)*vecf")
        else:  # correct
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            print(f"    theta={th/np.pi:.2f}pi phi={phi/np.pi:.2f}pi: n=({n[0]:+.0f},{n[1]:+.0f},{n[2]:+.0f}) gamma=n x phi=({gam[0]:+.2f},{gam[1]:+.2f},{gam[2]:+.2f})")
            ops.append(ox*Opy(th,svec(gam/gn)) if gn>1e-9 else ox)
        vecf=rot_xy(2*th,phi)*vecf*rot_xy(2*th,phi).dag()
        n=rot_bloch(n,phiv,th)
    return ops
def Ufull(ops):
    U=ops[0]
    for o in ops[1:]: U=o*U
    return U

N=41; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

P={}; UU={}
for mode in ['bare','old','correct','ideal']:
    ops=build(mode); P[mode]=sweepP(ops); UU[mode]=(Ufull(ops).dag()*Ufull(ops)-I2).norm()
pid=P['ideal']; mask=np.abs(pid-np.round(pid))<0.07; tgt=np.round(pid)
print("\nmode      op-err   ||U^dU-I||")
for mode in ['bare','old','correct','ideal']:
    e=np.mean(np.abs(P[mode]-tgt)[mask])
    print(f"  {mode:<8} {e:.4f}   {UU[mode]:.1e}")
print(f"\nmax |P_old - P_correct| over sweep = {np.max(np.abs(P['old']-P['correct'])):.3f}")

plt.figure(figsize=(9,5.5))
for mode,c,ls in [('bare','cornflowerblue','-'),('old','gray','-.'),('correct','seagreen','-'),('ideal','firebrick','--')]:
    e=np.mean(np.abs(P[mode]-tgt)[mask])
    plt.plot(m,P[mode],ls,color=c,lw=2.4 if mode in('correct','ideal') else 2,label=f"{mode} (op {e:.3f})")
plt.xlabel(r"$\langle x\rangle/\sqrt{\pi}$"); plt.ylabel("P(+1)")
plt.title("BB1(GCR) physical: OLD (wrong-sign) axis vs CORRECT axis $\\hat n\\times\\hat\\phi$")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Paper_Figures/bb1_axiscompare.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/bb1_axiscompare.png")
