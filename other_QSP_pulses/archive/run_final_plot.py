"""
Final BB1(GCR) figure with the requested colors, all on one grid (Ncav=120, N=81).
  bare BB1                          : blue solid
  BB1(GCR) (notebook, non-unitary)  : red solid   (should coincide with ideal)
  ideal (all vec*vecf)              : red dashed
  BB1(GCR) physical, no fix         : gray dotted
  fix: split pre-correction (K=12)  : purple solid
Prints P(+1) at integer m (plateaus) so we can see whether the fix beats bare there.
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
def sigma_xyz(th,phi): return np.cos(th)*SZ+np.sin(th)*sigma(phi)
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def vec_f(v,R): return R*v*R.dag()
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
theta=np.pi/2; phi0=0; phi1=np.arccos(-1/8); x=np.pi/theta; beta=theta/2/(np.sqrt(np.pi)/2)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]

# notebook GCR_BB1 operators (input-independent) — exact cell 8f78613c
def notebook_ops():
    vecf=vec_f(sigma_xyz(0,0),rot_xy(0,0)); vec=1j*sigma(phi1)*vecf
    O4=Opx(np.pi,phi1)*tensor(-1j*x*beta*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(3*phi1)*vecf
    O3=Opx(2*np.pi,3*phi1)*tensor(-2j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(4*np.pi,3*phi1)); vec=1j*sigma(phi1)*vecf
    O2=Opx(np.pi,phi1)*tensor(-1j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(phi0)*vecf
    O1=Opx(np.pi/2,phi0)*tensor(-1j*beta*(ep*pOp),vec).expm(method='dense')
    return [O4,O3,O2,O1]
def build(mode,seq=BB1):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:
            gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
def corr_split(K):
    out=[]
    for i,(th,phi) in enumerate(BB1): out+=[(th,phi)] if i==3 else [(th/K,phi)]*K
    return out

OPS={'bare':build('bare'),'gcr':notebook_ops(),'ideal':build('ideal'),
     'phys':build('phys'),'split':build('phys',corr_split(12))}
print(f"[{time.time()-t0:.0f}s] built ops")

N=81; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
def sweep(ops):
    out=[]
    for al in alph:
        it=(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
P={k:sweep(v) for k,v in OPS.items()}
print(f"[{time.time()-t0:.0f}s] swept")

print("\nP(+1) at integer m (plateaus; ideal target 1,0,1,0,1):")
print(f"{'m':>6}{'bare':>8}{'BB1(GCR)':>10}{'split-fix':>11}{'phys-nofix':>12}")
for i in range(N):
    if abs(m[i]-round(m[i]))<1e-6:
        print(f"{m[i]:>6.0f}{P['bare'][i]:>8.3f}{P['gcr'][i]:>10.3f}{P['split'][i]:>11.3f}{P['phys'][i]:>12.3f}")
# fix vs bare deviation from ideal square wave at plateaus
tgt=np.round(P['ideal']); pl=np.abs(m-np.round(m))<1e-6
print(f"\nmean |P-target| at plateaus:  bare={np.mean(np.abs(P['bare']-tgt)[pl]):.4f}"
      f"  split-fix={np.mean(np.abs(P['split']-tgt)[pl]):.4f}"
      f"  BB1(GCR)={np.mean(np.abs(P['gcr']-tgt)[pl]):.4f}")

np.savez("Paper_Data/final_plot.npz",m=m,**P)
plt.figure(figsize=(10,6))
plt.plot(m,P['bare'],color='tab:blue',lw=2.0,label='bare BB1')
plt.plot(m,P['gcr'],color='firebrick',lw=2.4,label='BB1(GCR), notebook (non-unitary)')
plt.plot(m,P['ideal'],color='firebrick',lw=1.6,ls='--',label='ideal ($\\sigma_\\gamma\\!\\to\\!i\\sigma_\\phi$)')
plt.plot(m,P['phys'],color='0.55',lw=2.0,ls=':',label='BB1(GCR), physical (no fix)')
plt.plot(m,P['split'],color='purple',lw=2.0,label='fix: split pre-correction (K=12)')
plt.xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); plt.ylabel('P(+1)')
plt.title('BB1(GCR) modular readout')
plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Paper_Figures/final_plot.png",dpi=130,bbox_inches="tight")
plt.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/final_plot.png and Supp_Figures/BB1_GCR_fixes.pdf")
