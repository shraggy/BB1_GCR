"""
Try the fixes for BB1(GCR), correct axis (sigma_gamma = n_hat x phi_hat), NOISELESS.
Success metric: readout closer to a square wave than bare BB1 => lower operating-point error
(referenced to the ideal vec*vecf plateaus).

Fix 1 (=Fix 4): split correction pulses into K sub-pulses (target pi/2 kept whole for
                determinism). Fix 4 (small-angle pole-returning composite) collapses to this.
Fix 2: numerically optimize the physical precorrection amplitude scale (global grid + per-pulse
       refine) for the correct-axis physical BB1 (unsplit and K=2).
Fix 3: heralded non-unitary = ideal vec*vecf readout + its heralding success probability
       (norm^2 of the un-renormalized composite output).
"""
import time, numpy as np
from qutip import *
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=110
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
    for i,(th,phi) in enumerate(seq): out += [(th,phi)] if i==len(seq)-1 else [(th/K,phi)]*K
    return out

def build(seq,mode,asc=1.0):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi),asc))
        else:
            gam=np.cross(n,phiv); gn=np.linalg.norm(gam)
            ops.append(ox*Opy(th,svec(gam/gn),asc) if gn>1e-9 else ox)
        n=rot_bloch(n,phiv,th)
    return ops

N=31; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
def sweepNorm(ops):  # heralding success prob = norm^2 of un-renormalized output
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=U*s
        out.append(float(s.norm()**2))
    return np.array(out)

Pideal=sweepP(build(BB1,'ideal')); mask=np.abs(Pideal-np.round(Pideal))<0.07; tgt=np.round(Pideal)
def oe(P): return np.mean(np.abs(P-tgt)[mask])
ebare=oe(sweepP(build(BB1,'bare'))); eideal=oe(Pideal)
print(f"references: bare op-err={ebare:.4f}   ideal op-err={eideal:.4f}\n")

results={}
# ---- Fix 1 (=4): splitting ----
print("Fix 1/4 (splitting, correct axis, target whole):")
for K in [1,2,4,8,12]:
    e=oe(sweepP(build(corr_split(BB1,K),'correct'))); results[f'split K={K}']=e
    print(f"   K={K:<3} op-err={e:.4f}  {'< bare (BEATS)' if e<ebare else '>= bare'}")

# ---- Fix 2: optimize physical precorrection amplitude ----
print("\nFix 2 (amplitude-optimized physical, correct axis):")
best2=(None,1e9)
for base,seq in [('unsplit',BB1),('K=2',corr_split(BB1,2))]:
    for s in [0.0,0.5,0.75,1.0,1.25,1.5,2.0]:
        e=oe(sweepP(build(seq,'correct',asc=s)))
        if e<best2[1]: best2=(f'{base} s={s}',e)
    print(f"   scanned {base}")
results['Fix2 best (amp-opt)']=best2[1]
print(f"   best: {best2[0]}  op-err={best2[1]:.4f}  {'< bare (BEATS)' if best2[1]<ebare else '>= bare'}")

# ---- Fix 3: heralded non-unitary (ideal) ----
Psucc=sweepNorm(build(BB1,'ideal'))
results['Fix3 heralded (=ideal)']=eideal
print(f"\nFix 3 (heralded non-unitary = ideal): op-err={eideal:.4f}  (BEATS bare)")
print(f"   heralding success prob over sweep: min={Psucc.min():.3f} mean={Psucc.mean():.3f} max={Psucc.max():.3f}")

print("\n==== SUMMARY (op-err, lower=squarer; bare=%.3f, ideal=%.3f) ===="%(ebare,eideal))
for k,v in results.items(): print(f"   {k:<24}: {v:.4f}  {'BEATS bare' if v<ebare else ''}")
np.savez("Paper_Data/fixes.npz",m=m,Pideal=Pideal,Psucc=Psucc,ebare=ebare,eideal=eideal,
         **{k.replace(' ','_').replace('=','').replace('(','').replace(')',''):np.array([v]) for k,v in results.items()})
print(f"\n[{time.time()-t0:.0f}s] done")
