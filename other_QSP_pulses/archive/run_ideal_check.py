"""
Verify the IDEAL (vec*vecf) within the CORRECT-axis framework for BB1.
Claims to check:
  1. vec*vecf = 1j sigma(phi) exactly, independent of the tracked axis n_hat (so the ideal is
     the same whether or not the physical axis is constructed correctly).
  2. Applying vec*vecf gives the expected good ideal-case readout (well below bare), and we
     report it robustly at the bin centers (extrema), not just the plateau-mask average.
"""
import time, numpy as np
from qutip import *
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

# --- Claim 1: vec*vecf = 1j sigma(phi) regardless of n_hat ---
print("=== vec*vecf vs 1j*sigma(phi) along BB1's ACTUAL n_hat trajectory ===")
n=np.array([0,0,1.0])
for th,phi in BB1:
    vecf=svec(n)                      # sigma_{n_hat}  (the tracked qubit axis)
    vec=1j*sigma(phi)*vecf            # notebook's vec
    diff=(vec*vecf - 1j*sigma(phi)).norm()
    print(f"  theta={th/np.pi:.2f}pi n=({n[0]:+.0f},{n[1]:+.0f},{n[2]:+.0f}): ||vec*vecf - 1j sigma(phi)|| = {diff:.2e}")
    n=rot_bloch(n,np.array([np.cos(phi),np.sin(phi),0.0]),th)

def build(mode):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in BB1:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            ops.append(ox*Opy(th,svec(gam/gn)) if gn>1e-9 else ox)
        n=rot_bloch(n,phiv,th)
    return ops

N=81; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    return np.array([ (lambda s:np.real(expect(ket2dm(py),s.ptrace(1))))(_r(ops,it)) for it in inits])
def _r(ops,it):
    s=tensor(it,g)
    for U in ops: s=(U*s).unit()
    return s

P={m_:sweepP(build(m_)) for m_ in ['bare','correct','ideal']}
pid=P['ideal']; mask=np.abs(pid-np.round(pid))<0.07; tgt=np.round(pid)
print("\n=== BB1 readout error (correct-axis framework), N=81 ===")
print(f"{'mode':<9} {'op-err(plateau)':>16} {'worst@bin-center':>18}")
for m_ in ['bare','correct','ideal']:
    ope=np.mean(np.abs(P[m_]-tgt)[mask])
    wc=np.max(np.abs(P[m_]-tgt)[mask])
    print(f"{m_:<9} {ope:>16.4f} {wc:>18.4f}")
# ideal readout contrast at the central plateaus (should be ~0 and ~1)
print(f"\nideal P(+1): max over sweep={P['ideal'].max():.4f}  min={P['ideal'].min():.4f}")
print(f"bare  P(+1): max={P['bare'].max():.4f}  min={P['bare'].min():.4f}")
print(f"\n[{time.time()-t0:.0f}s] done")
