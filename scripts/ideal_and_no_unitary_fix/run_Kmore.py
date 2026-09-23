"""Compute larger K (32,48) for the green (correct-axis) split fix; combine with K=1..16 from
Kconv.npz. Plot full-range P(+1) (left) + log P(-1) near x=0 (right) with each K labeled in its
own green shade. Question: does more splitting reach the dashed ideal, or plateau at bare?"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def corr_split(K):
    out=[]
    for i,(th,phi) in enumerate(BB1): out+=[(th,phi)] if i==3 else [(th/K,phi)]*K
    return out
def build_split(K):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in corr_split(K):
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
d=np.load("Paper_Data/Kconv.npz"); m=d["m"]
alph=m*sq; a=-sq/2
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
C={k:d[k] for k in d.files if k!="m"}
for K in [32,48]:
    C[f'K{K}']=sweep(build_split(K)); print(f"[{time.time()-t0:.0f}s] K={K} done")
np.savez("Paper_Data/Kconv.npz",m=m,**C)
i0=int(np.argmin(np.abs(m)))
Ks=[1,2,4,8,16,32,48]
print("\n K :  P(-1)@x=0")
print(f"ideal: {1-C['ideal'][i0]:.2e}"); print(f"bare : {1-C['bare'][i0]:.4f}")
for K in Ks: print(f" K={K:<3}: {1-C[f'K{K}'][i0]:.4f}")

greens=plt.cm.Greens(np.linspace(0.32,1.0,len(Ks)))
def clip(v): return np.clip(v,1e-7,None)
fig=plt.figure(figsize=(12.8,5.4)); gs=GridSpec(1,2,width_ratios=[2.0,1.0],wspace=0.24)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])
for j,K in enumerate(Ks): ax.plot(m,C[f'K{K}'],color=greens[j],lw=1.6)
ax.plot(m,C['ideal'],color='firebrick',lw=2.2,ls='--',label='BB1(GCR) ideal (non-unitary)',zorder=9)
ax.plot(m,C['bare'],color='dodgerblue',lw=3.4,label='bare BB1',zorder=10)
ax.plot([],[],color=greens[-1],lw=2,label='physical pulse, $K$ splits (green)')
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.3)
ax.set_title('BB1(GCR) modular readout (full range)'); ax.legend(fontsize=8.5,loc='center')
ax.axvspan(-0.5,0.5,color='0.93',zorder=0)
for j,K in enumerate(Ks): axr.plot(m,clip(1-C[f'K{K}']),color=greens[j],lw=1.6)
axr.plot(m,clip(1-C['ideal']),color='firebrick',lw=2.2,ls='--',zorder=9)
axr.plot(m,clip(1-C['bare']),color='dodgerblue',lw=3.4,zorder=10)
# label each K in its color, at x=-0.42
xl=-0.42; il=int(np.argmin(np.abs(m-xl)))
for j,K in enumerate(Ks):
    axr.text(xl, clip(1-C[f'K{K}'])[il]*1.15, f'K={K}', color=greens[j], fontsize=8, fontweight='bold', ha='center')
axr.set_yscale('log'); axr.set_xlim(-0.5,0.5); axr.set_ylim(1e-6,1)
axr.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axr.set_ylabel(r'$P(-1)$')
axr.set_title(r'$P(-1)$ near $\langle x\rangle=0$ (log)'); axr.grid(alpha=0.3)
fig.suptitle(r'Green = physical pulse with $K$ splits; does it reach the ideal (dashed)?',fontsize=12)
fig.tight_layout()
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
fig.savefig("Paper_Figures/Kfull.png",dpi=130,bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved.")
