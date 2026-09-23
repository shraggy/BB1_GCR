"""Re-plot from saved curves: left = P(+1) full range; right = P(-1) vs x NEAR x=0 on log scale."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
d=np.load("Paper_Data/final_plot2.npz"); m=d["m"]
cur={k:d[k] for k in d.files if k!="m"}
def Pm1(P): return np.clip(1.0-P,1e-7,None)

fig=plt.figure(figsize=(12,5.2))
gs=GridSpec(1,2,width_ratios=[1.9,1.0],wspace=0.28)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])
grays={'K4':'0.68','K8':'0.5','K12':'0.28'}

# left: P(+1) full
ax.plot(m,cur['bare'],color='tab:blue',lw=2.0,label='bare BB1')
ax.plot(m,cur['ideal'],color='firebrick',lw=1.8,ls='--',label='BB1(GCR) ideal (non-unitary)')
ax.plot(m,cur['phys'],color='firebrick',lw=2.4,label='BB1(GCR) physical, no fix')
for K in [4,8,12]: ax.plot(m,cur[f'K{K}'],color=grays[f'K{K}'],lw=1.8,label=f'fix: split K={K} (unitary)')
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.3)
ax.legend(fontsize=8,loc='center'); ax.set_title('BB1(GCR) modular readout')

# right: P(-1) near x=0, log scale
ax.axvspan(-0.7,0.7,color='0.9',alpha=0.5,zorder=0)
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
print("P(-1) at x=0:", {k:float(Pm1(cur[k])[int(np.argmin(np.abs(m)))]) for k in ['ideal','bare','K12','K8','K4','phys']})
print("saved Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/final_plot2.png")
