"""Full-range P(+1) (left) + log P(-1) near x=0 (side), showing the green (correct-axis)
split-fix convergence K=1..16 vs bare BB1 and the ideal. Reuses Paper_Data/Kconv.npz."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
d=np.load("Paper_Data/Kconv.npz"); m=d["m"]
Ks=[1,2,4,8,12,16]; greens=plt.cm.Greens(np.linspace(0.35,1.0,len(Ks)))
def clip(v): return np.clip(v,1e-7,None)

fig=plt.figure(figsize=(12.5,5.2)); gs=GridSpec(1,2,width_ratios=[2.0,1.0],wspace=0.26)
ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])

# main: full-range P(+1)
for j,K in enumerate(Ks):
    ax.plot(m,d[f'K{K}'],color=greens[j],lw=1.7,label=(f'physical no fix (K=1)' if K==1 else f'split fix K={K}'))
ax.plot(m,d['ideal'],color='firebrick',lw=2.2,ls='--',label='BB1(GCR) ideal (non-unitary)',zorder=9)
ax.plot(m,d['bare'],color='dodgerblue',lw=3.4,label='bare BB1',zorder=10)
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.3)
ax.set_title('BB1(GCR) modular readout (full range)'); ax.legend(fontsize=8,loc='center',ncol=2)
ax.axvspan(-0.5,0.5,color='0.93',zorder=0)

# side: log P(-1) near x=0
for j,K in enumerate(Ks):
    axr.plot(m,clip(1-d[f'K{K}']),color=greens[j],lw=1.7)
axr.plot(m,clip(1-d['ideal']),color='firebrick',lw=2.2,ls='--',zorder=9)
axr.plot(m,clip(1-d['bare']),color='dodgerblue',lw=3.4,zorder=10)
axr.set_yscale('log'); axr.set_xlim(-0.5,0.5); axr.set_ylim(1e-6,1)
axr.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axr.set_ylabel(r'$P(-1)$')
axr.set_title(r'$P(-1)$ near $\langle x\rangle=0$ (log)'); axr.grid(alpha=0.3)
fig.suptitle(r'Correct-axis split fix converges to bare BB1 (never to the ideal) as $K$ grows',fontsize=12)
fig.tight_layout()
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
fig.savefig("Paper_Figures/Kfull.png",dpi=130,bbox_inches="tight")
print("saved Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/Kfull.png")
