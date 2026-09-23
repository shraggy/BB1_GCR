"""
(a) Fix 3 heralding success probability via proper Kraus normalization.
(b) Plot P(+1) square waves: bare, Fix 1/4 (split K=12), Fix 2 (per-pulse amplitude-optimized
    physical), ideal (=Fix 3). Correct axis, noiseless.
"""
import time, numpy as np
from qutip import *
from scipy.optimize import minimize
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
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis,asc=1.0): return tensor(-1j*asc*(th*ep/sq)*pOp,axis).expm(method='dense')
phi1=np.arccos(-1/8)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def corr_split(seq,K):
    out=[]
    for i,(th,phi) in enumerate(seq): out+=[(th,phi)] if i==len(seq)-1 else [(th/K,phi)]*K
    return out

def build(seq,mode,ascales=None):
    """ascales: per-pulse amplitude scale (aligned with seq)."""
    ops=[]; n=np.array([0,0,1.0])
    for j,(th,phi) in enumerate(seq):
        asc=1.0 if ascales is None else ascales[j]
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi),asc))
        else:
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            ops.append(ox*Opy(th,svec(gam/gn),asc) if gn>1e-9 else ox)
        n=rot_bloch(n,phiv,th)
    return ops

N=41; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
Pideal=sweepP(build(BB1,'ideal')); mask=np.abs(Pideal-np.round(Pideal))<0.07; tgt=np.round(Pideal)
def oe(P): return np.mean(np.abs(P-tgt)[mask])
Pbare=sweepP(build(BB1,'bare')); ebare=oe(Pbare); eideal=oe(Pideal)
print(f"[{time.time()-t0:.0f}s] bare={ebare:.4f} ideal={eideal:.4f}")

# ---- Fix 1/4: split K=12 ----
Psplit=sweepP(build(corr_split(BB1,12),'correct')); esplit=oe(Psplit)
print(f"[{time.time()-t0:.0f}s] Fix1/4 split K=12: op-err={esplit:.4f}")

# ---- Fix 2: per-pulse amplitude optimization (correct-axis physical, unsplit) ----
def obj(x):  # x = 4 amplitude scales
    return oe(sweepP(build(BB1,'correct',ascales=list(x))))
res=minimize(obj,x0=np.ones(4),method='Nelder-Mead',
             options={'maxfev':80,'xatol':1e-2,'fatol':1e-3})
Pfix2=sweepP(build(BB1,'correct',ascales=list(res.x))); efix2=oe(Pfix2)
print(f"[{time.time()-t0:.0f}s] Fix2 opt amps={np.round(res.x,3)} op-err={efix2:.4f}")

# ---- Fix 3: heralding success probability via Kraus normalization ----
# raw non-unitary composite M (product of ideal stages, no renorm)
ops_ideal=build(BB1,'ideal')
M=ops_ideal[0]
for o in ops_ideal[1:]: M=o*M
lam=np.linalg.svd(M.full(),compute_uv=False)[0]   # largest singular value
psucc=[]
for it in inits:
    v=(M*tensor(it,g)).norm()**2
    psucc.append(v/lam**2)
psucc=np.array(psucc)
print(f"[{time.time()-t0:.0f}s] Fix3 heralding: ||M||_op(lambda_max)={lam:.3f}")
print(f"   success prob = ||M psi||^2 / lambda_max^2 : min={psucc.min():.3e} mean={psucc.mean():.3e} max={psucc.max():.3e}")

# ---- plot ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(m,Pbare,color='cornflowerblue',lw=2,label=f'bare BB1 (op {ebare:.3f})')
plt.plot(m,Psplit,color='seagreen',lw=2.4,label=f'Fix 1/4 split K=12 (op {esplit:.3f})')
plt.plot(m,Pfix2,color='darkorange',lw=2,ls='-.',label=f'Fix 2 amp-optimized physical (op {efix2:.3f})')
plt.plot(m,Pideal,color='firebrick',lw=2.6,ls='--',label=f'ideal = Fix 3 heralded (op {eideal:.3f})')
plt.xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); plt.ylabel('P(+1)')
plt.title('BB1(GCR) fixes: readout square wave (correct axis, noiseless)')
plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Paper_Figures/fixes_squarewave.png",dpi=130,bbox_inches="tight")
np.savez("Paper_Data/fixes_final.npz",m=m,Pbare=Pbare,Psplit=Psplit,Pfix2=Pfix2,Pideal=Pideal,
         psucc=psucc,lam=lam,ebare=ebare,esplit=esplit,efix2=efix2,eideal=eideal)
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/fixes_squarewave.png")
print(f"\nSUMMARY op-err (bare={ebare:.3f}): split-K12={esplit:.3f}  Fix2={efix2:.3f}  ideal/Fix3={eideal:.3f}")
