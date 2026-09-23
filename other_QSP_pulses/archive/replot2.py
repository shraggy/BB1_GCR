"""Red solid = your notebook 'vec' physical no-fix (axis 1j*sigma(phi)*vecf, = sigma(phi+90) on
all pulses for BB1) -- matches the gray curve in bb1_axiscompare.png.
Right panel: log P(-1) vs x near x=0 (not bars). Reuse bare/ideal/splits from final_plot2.npz."""
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
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def vec_f(v,R): return R*v*R.dag()
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
# notebook 'vec' physical no-fix: replace vec*vecf -> vec everywhere, notebook frame tracking (2theta)
def phys_notebook():
    ops=[]; vecf=SZ
    for th,phi in [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]:
        vec=1j*sigma(phi)*vecf
        ops.append(Opx(th,phi)*Opy(th,vec))
        vecf=vec_f(vecf,rot_xy(2*th,phi))
    return ops
N=81; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
phys=sweep(phys_notebook())
print(f"[{time.time()-t0:.0f}s] notebook-vec physical no-fix computed")

d=np.load("Paper_Data/final_plot2.npz")
cur={k:d[k] for k in d.files if k!="m"}; cur['phys']=phys
np.savez("Paper_Data/final_plot2.npz",m=m,**cur)
def Pm1(P): return np.clip(1.0-P,1e-7,None)
grays={'K4':'0.68','K8':'0.5','K12':'0.28'}
fig=plt.figure(figsize=(12,5.2)); gs=GridSpec(1,2,width_ratios=[1.9,1.0],wspace=0.28)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])
ax.plot(m,cur['bare'],color='tab:blue',lw=2.0,label='bare BB1')
ax.plot(m,cur['ideal'],color='firebrick',lw=1.8,ls='--',label='BB1(GCR) ideal (non-unitary)')
ax.plot(m,cur['phys'],color='firebrick',lw=2.4,label='BB1(GCR) physical, no fix (vec)')
for K in [4,8,12]: ax.plot(m,cur[f'K{K}'],color=grays[f'K{K}'],lw=1.8,label=f'fix: split K={K} (unitary)')
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.3)
ax.legend(fontsize=8,loc='center'); ax.set_title('BB1(GCR) modular readout')
ax.axvspan(-0.7,0.7,color='0.92',zorder=0)
axr.plot(m,Pm1(cur['bare']),color='tab:blue',lw=2.0)
axr.plot(m,Pm1(cur['ideal']),color='firebrick',lw=1.8,ls='--')
axr.plot(m,Pm1(cur['phys']),color='firebrick',lw=2.4)
for K in [4,8,12]: axr.plot(m,Pm1(cur[f'K{K}']),color=grays[f'K{K}'],lw=1.8)
axr.set_yscale('log'); axr.set_xlim(-0.7,0.7); axr.set_ylim(1e-5,1)
axr.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axr.set_ylabel(r'$P(-1)$')
axr.grid(alpha=0.3); axr.set_title(r'$P(-1)$ near $\langle x\rangle=0$ (log)')
fig.tight_layout()
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
fig.savefig("Paper_Figures/final_plot2.png",dpi=130,bbox_inches="tight")
i0=int(np.argmin(np.abs(m)))
print("P(-1) at x=0:", {k:float(Pm1(cur[k])[i0]) for k in ['ideal','bare','K12','K8','K4','phys']})
print(f"[{time.time()-t0:.0f}s] saved.")
