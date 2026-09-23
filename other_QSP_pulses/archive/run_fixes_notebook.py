"""
Apply the 4 fixes to the RIGHT (notebook) BB1(GCR) pulse:
 reference = exact notebook GCR_BB1 (vec*vecf on the 3 corrections, vec on the final target).
Judge by P(+1) at the operating points (integer m = <x>/sqrt(pi)); bare and notebook are refs.
 Fix1/4 splitting: replace the 3 non-unitary corrections with split PHYSICAL sub-pulses (K), keep target.
 Fix2 amplitude-opt: physical corrections, scan a global amplitude scale.
 Fix3 heralding: the notebook pulse IS non-unitary; success prob = ||M psi||^2 / lambda_max(M)^2.
"""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def sigma_xyz(th,phi): return np.cos(th)*SZ+np.sin(th)*sigma(phi)
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def vec_f(v,R): return R*v*R.dag()
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
theta=np.pi/2; phi0=0; phi1=np.arccos(-1/8); x=np.pi/theta; beta=theta/2/(np.sqrt(np.pi)/2)
I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis,asc=1.0): return tensor(-1j*asc*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]

def notebook_ops():   # exact cell 8f78613c
    vecf=vec_f(sigma_xyz(0,0),rot_xy(0,0)); vec=1j*sigma(phi1)*vecf
    O4=Opx(np.pi,phi1)*tensor(-1j*x*beta*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(3*phi1)*vecf
    O3=Opx(2*np.pi,3*phi1)*tensor(-2j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(4*np.pi,3*phi1)); vec=1j*sigma(phi1)*vecf
    O2=Opx(np.pi,phi1)*tensor(-1j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(phi0)*vecf
    O1=Opx(np.pi/2,phi0)*tensor(-1j*beta*(ep*pOp),vec).expm(method='dense')
    return [O4,O3,O2,O1]
def build_phys(seq,asc=1.0):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam)),asc))
        n=rot_bloch(n,phiv,th)
    return ops
def build_bare(seq):
    return [Opx(th,phi) for th,phi in seq]
def corr_split(K):
    out=[]
    for i,(th,phi) in enumerate(BB1): out+=[(th,phi)] if i==3 else [(th/K,phi)]*K
    return out

N=41; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
def sweepNorm(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=U*s
        out.append(float(s.norm()**2))
    return np.array(out)

nb=notebook_ops()
P_nb=sweepP(nb); P_bare=sweepP(build_bare(BB1)); P_split=sweepP(build_phys(corr_split(12)))
tgt=np.round(P_nb); pl=np.abs(m-np.round(m))<1e-6
def plat(P): return np.mean(np.abs(P-tgt)[pl])
print(f"[{time.time()-t0:.0f}s] plateau mean|P-target|:  bare={plat(P_bare):.4f}  notebook(BB1(GCR))={plat(P_nb):.4f}  Fix1/4 split-K12={plat(P_split):.4f}")

# Fix2: amplitude scan on physical corrections
best=(None,9);
for asc in [0.0,0.5,1.0,1.5]:
    e=plat(sweepP(build_phys(BB1,asc)))
    if e<best[1]: best=(asc,e)
print(f"[{time.time()-t0:.0f}s] Fix2 amplitude-opt physical: best scale s={best[0]} -> plateau err={best[1]:.4f}")

# Fix3: heralding cost of the exact notebook pulse
M=nb[0]
for o in nb[1:]: M=o*M
lam=np.linalg.svd(M.full(),compute_uv=False)[0]
nrm=sweepNorm(nb)
print(f"[{time.time()-t0:.0f}s] Fix3 heralding of the notebook pulse:")
print(f"   ||M psi||^2 (norm amplification on relevant inputs): min={nrm.min():.2f} mean={nrm.mean():.2f}")
print(f"   lambda_max(M)={lam:.2f} (truncation-limited);  success = ||M psi||^2/lambda_max^2: mean={nrm.mean()/lam**2:.2e}")
print(f"\n[{time.time()-t0:.0f}s] done")
