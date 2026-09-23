"""Final appendix figure, full range x/sqrt(pi) in [-2,2].
 Red solid = physical no-fix with the CORRECT axis sigma_gamma = n_hat x phi_hat (confirmed right).
 bare=blue, ideal=red dashed, split fixes K=4,8,12 = gray shades. Right panel: log P(-1) near x=0.
 Reuses split curves (correct-axis) from final_plot2.npz; recomputes the correct-axis no-fix."""
import numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def phys_correct():
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in BB1:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
d=np.load("Paper_Data/final_plot2.npz"); m=d["m"]
alph=m*sq; a=-sq/2
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
s=phys_correct(); phys=[]
for it in inits:
    st=tensor(it,g)
    for U in s: st=(U*st).unit()
    phys.append(np.real(expect(ket2dm(py),st.ptrace(1))))
phys=np.array(phys)
cur={'bare':d['bare'],'ideal':d['ideal'],'phys':phys,'K4':d['K4'],'K8':d['K8'],'K12':d['K12']}
np.savez("Paper_Data/final_plot2.npz",m=m,**cur)
def Pm1(P): return np.clip(1.0-P,1e-7,None)
grays={'K4':'0.68','K8':'0.5','K12':'0.28'}
fig=plt.figure(figsize=(12,5.2)); gs=GridSpec(1,2,width_ratios=[1.9,1.0],wspace=0.28)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])
ax.plot(m,cur['bare'],color='tab:blue',lw=2.0,label='bare BB1')
ax.plot(m,cur['ideal'],color='firebrick',lw=1.8,ls='--',label='BB1(GCR) ideal (non-unitary)')
ax.plot(m,cur['phys'],color='firebrick',lw=2.4,label=r'BB1(GCR) physical, no fix ($\hat n\!\times\!\hat\phi$)')
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
print("phys(correct) P(-1)@x=0 =",round(float(1-phys[i0]),4))
print("saved Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/final_plot2.png")
