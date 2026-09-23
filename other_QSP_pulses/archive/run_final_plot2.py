"""
BB1(GCR) figure, requested layout:
  (1) bare BB1                          : blue solid
  (2) ideal (non-unitary BB1(GCR))      : red DASHED  (== notebook pulse; not replotted separately)
  (3) BB1(GCR) physical, no fix         : red SOLID   (same red, physical realization)
  (4) unitary fixes (splitting, K)      : shades of gray
Inset: log-scale P(-1) at x=0 (the operating point) for each scheme, to highlight the error.
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
phi1=np.arccos(-1/8)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
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

N=81; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

curves={}
curves['bare']=sweep(build('bare'))
curves['ideal']=sweep(build('ideal'))
curves['phys']=sweep(build('phys'))
for K in [4,8,12]:
    curves[f'K{K}']=sweep(build('phys',corr_split(K)))
    print(f"[{time.time()-t0:.0f}s] split K={K} done")
i0=int(np.argmin(np.abs(m)))   # index of x=0
np.savez("Paper_Data/final_plot2.npz",m=m,**curves)

# P(-1) at x=0
Pm1={k:max(1-curves[k][i0],1e-6) for k in curves}
print("\nP(-1) at x=0:")
for k in ['ideal','bare','K12','K8','K4','phys']:
    print(f"   {k:>6}: {Pm1[k]:.3e}")

fig=plt.figure(figsize=(12,5.2))
gs=GridSpec(1,3,width_ratios=[2.3,0.05,1.0],wspace=0.05)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[2])
grays={'K4':'0.68','K8':'0.5','K12':'0.28'}
ax.plot(m,curves['bare'],color='tab:blue',lw=2.0,label='bare BB1')
ax.plot(m,curves['ideal'],color='firebrick',lw=1.8,ls='--',label='BB1(GCR) ideal (non-unitary)')
ax.plot(m,curves['phys'],color='firebrick',lw=2.4,label='BB1(GCR) physical, no fix')
for K in [4,8,12]:
    ax.plot(m,curves[f'K{K}'],color=grays[f'K{K}'],lw=1.8,label=f'fix: split K={K} (unitary)')
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel('P(+1)'); ax.grid(alpha=0.3)
ax.legend(fontsize=8,loc='center'); ax.set_title('BB1(GCR) modular readout')

order=['ideal','bare','K12','K8','K4','phys']
cols={'ideal':'firebrick','bare':'tab:blue','K12':grays['K12'],'K8':grays['K8'],'K4':grays['K4'],'phys':'firebrick'}
axr.bar(range(len(order)),[Pm1[k] for k in order],
        color=[cols[k] for k in order],edgecolor='k',lw=0.5)
axr.set_yscale('log'); axr.set_ylabel(r'$P(-1)$ at $\langle x\rangle=0$')
axr.set_xticks(range(len(order)))
axr.set_xticklabels(['ideal','bare','K12','K8','K4','phys\nno-fix'],rotation=45,fontsize=8,ha='right')
axr.grid(alpha=0.3,axis='y'); axr.set_title('operating-point error (log)',fontsize=10)
fig.tight_layout()
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
fig.savefig("Paper_Figures/final_plot2.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/final_plot2.png")
