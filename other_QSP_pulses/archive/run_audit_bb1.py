"""
AUDIT: recompute every 'BB1(GCR) physical' quantity reported earlier using the CORRECT axis
(sigma_gamma = n_hat x phi_hat, n_hat from no-error trajectory) and compare to the OLD
(wrong-axis) numbers. Checks whether any conclusion changes. BB1 readout, net theta=pi/2.
Metric: operating-point error referenced to each config's ideal plateaus; + unitarity.
"""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=120
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,axis,angle):
    k=np.array(axis,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(angle)+np.cross(k,n)*np.sin(angle)+k*np.dot(k,n)*(1-np.cos(angle))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')

def build(seq, modes):
    """modes: per-pulse 'bare'/'phys'(correct axis)/'ideal'. n_hat from no-error rotations."""
    ops=[]; n=np.array([0,0,1.0])
    for (th,phi),mode in zip(seq,modes):
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            ops.append(ox*Opy(th,svec(gam/gn)) if gn>1e-9 else ox)
        n=rot_bloch(n,phiv,th)
    return ops
def Ufull(ops):
    U=ops[0]
    for o in ops[1:]: U=o*U
    return U

phi1=np.arccos(-1/8)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]        # idx0..3 = pi,2pi,pi,pi/2
BB1s=[(np.pi,phi1),(np.pi,3*phi1),(np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]  # 2pi->2xpi

N=25; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
Pideal=sweepP(build(BB1,['ideal']*4))
def oe(P):
    mask=np.abs(Pideal-np.round(Pideal))<0.07; return np.mean(np.abs(P-np.round(Pideal))[mask])

# (scheme name, seq, modes, OLD number reported earlier)
schemes=[
 ("bare",                     BB1,  ['bare']*4,                       0.041),
 ("ideal vec*vecf",           BB1,  ['ideal']*4,                      0.008),
 ("FULL physical BB1(GCR)",   BB1,  ['phys']*4,                       0.393),
 ("LOO: only pi(G4) phys",    BB1,  ['phys','ideal','ideal','ideal'], 0.236),
 ("LOO: only 2pi(G3) phys",   BB1,  ['ideal','phys','ideal','ideal'], 0.465),
 ("LOO: only pi(G2) phys",    BB1,  ['ideal','ideal','phys','ideal'], 0.322),
 ("LOO: only pi/2(G1) phys",  BB1,  ['ideal','ideal','ideal','phys'], 0.104),
 ("hybrid: 2pi bare, rest phys", BB1, ['phys','bare','phys','phys'],  0.297),
 ("2pi->2xpi, all phys",      BB1s, ['phys']*5,                       0.284),
]
print(f"{'scheme':<30}{'OLD(wrong)':>12}{'CORRECT':>10}{'||U^dU-I||':>13}  unit?")
rows=[]
for name,seq,modes,old in schemes:
    ops=build(seq,modes); P=sweepP(ops); e=oe(P); u=(Ufull(ops).dag()*Ufull(ops)-I2).norm()
    rows.append((name,old,e,u));
    print(f"{name:<30}{old:>12.3f}{e:>10.3f}{u:>13.1e}  {'U' if u<1e-6 else 'NU'}")
print(f"\n[{time.time()-t0:.0f}s] done.  (LOO = leave-one-out from the ideal chain; makes ONE stage physical.)")
print("Note: splitting convergence (correct axis) is in correct_split.png: BB1 corr-split K=8 = 0.044.")
