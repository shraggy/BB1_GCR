"""Re-plot the combined readout figure from saved data (Paper_Data/readout_combined.npz).
Panel (a): readout error vs eps (paper colors) -- 'BB1(GCR) ideal' (non-unitary limit, NOT called
post-selected: no explicit protocol is demonstrated for it) + 'BB1(GCR) heralded' (the demonstrated
heralded readout, ~17% herald pass) + Helstrom bound.
Panel (b): correct-readout probability P(g|eps)=1-P(e|eps), same curves; starts at ~1 near the peak.
The ~17%/~7% herald success probabilities are stated in the text, not plotted.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm',
                     'axes.labelsize':19,'xtick.labelsize':15,'ytick.labelsize':15,'axes.linewidth':3,
                     'xtick.direction':'in','ytick.direction':'in','xtick.major.width':2,'ytick.major.width':2,
                     'xtick.major.size':4,'ytick.major.size':4,'xtick.minor.width':1.5,'ytick.minor.width':1.5,
                     'xtick.minor.size':2.5,'ytick.minor.size':2.5,'legend.fontsize':12})
d=np.load("Paper_Data/readout_combined.npz"); eps=d['eps']; hel=float(d['helstrom'])
u=np.sqrt(np.pi)/(2*np.sqrt(2))   # displacement-error unit (= sqrt(pi/2)/2); peaks at even multiples of u
GOLD='goldenrod'; FIRE='firebrick'; ROY='royalblue'; FOR='forestgreen'; PUR='purple'
MSG='mediumseagreen'; DCY='darkcyan'
def clip(v): return np.clip(np.abs(v),1e-8,None)
fig=plt.figure(figsize=(12,5)); gs=GridSpec(1,2,width_ratios=[1.35,1.0],wspace=0.30)
ax=fig.add_subplot(gs[0]); axs=fig.add_subplot(gs[1])
def pn(a,l): a.text(-0.13,1.02,l,transform=a.transAxes,fontsize=20,fontweight='bold',va='bottom')
# (a) readout error
ax.plot(eps,clip(d['err_inf']),color=GOLD,lw=2,label='Infinite GKP')
ax.plot(eps,clip(d['err_gcr']),color=FIRE,lw=2,label='GCR')
ax.plot(eps,clip(d['err_bb1']),color=ROY,lw=2,label='BB1')
ax.plot(eps,clip(d['err_gcrbb1']),color=FOR,lw=2,label='GCR-BB1')
ax.plot(eps,clip(d['err_blockA']),color=MSG,lw=2,label='GCR-BB1 (block A)')
ax.plot(eps,clip(d['err_blockB']),color=DCY,lw=2,label='GCR-BB1 (block B)')
ax.plot(eps,clip(d['err_ideal']),'--',color=PUR,lw=2.4,label='BB1(GCR) ideal')
ax.plot(eps,clip(d['err_flag']),'-',color=PUR,lw=2.4,label='BB1(GCR) heralded (17% pass)')
ax.axhline(hel,ls=':',color='k',lw=2,label='Helstrom bound')
ax.set_yscale('log'); ax.set_xlim(eps[0],u); ax.set_ylim(1e-8,1)   # panel (a): logical-0 region [0,u]
ax.set_xticks([0,u]); ax.set_xticklabels([r"$0$",r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"])
ax.grid(alpha=0.35,which='both')
ax.set_xlabel(r'displacement error  $\epsilon$'); ax.set_ylabel(r'readout error  $P(e|\epsilon)$')
pn(ax,'(a)'); ax.legend(fontsize=10.5,loc='lower right',frameon=True,framealpha=0.92)
# (b) logical-1 readout error P(g|eps) = 1 - P(e|eps), log, over the logical-1 peak (eps in [1,1.6])
def one(v): return np.clip(1-np.abs(v),1e-8,None)
axs.plot(eps,one(d['err_inf']),color=GOLD,lw=2)
axs.plot(eps,one(d['err_gcr']),color=FIRE,lw=2)
axs.plot(eps,one(d['err_bb1']),color=ROY,lw=2)
axs.plot(eps,one(d['err_gcrbb1']),color=FOR,lw=2)
axs.plot(eps,one(d['err_blockA']),color=MSG,lw=2)
axs.plot(eps,one(d['err_blockB']),color=DCY,lw=2)
axs.plot(eps,one(d['err_ideal']),'--',color=PUR,lw=2.4)
axs.plot(eps,one(d['err_flag']),'-',color=PUR,lw=2.4)
axs.axhline(hel,ls=':',color='k',lw=2)                               # Helstrom bound (same as panel a)
axs.set_yscale('log'); axs.set_xlim(2*u,3*u); axs.set_ylim(1e-8,1); axs.grid(alpha=0.35,which='both')
axs.set_xticks([2*u,3*u]); axs.set_xticklabels([r"$\frac{\sqrt{\pi}}{\sqrt{2}}$",r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"])
axs.set_xlabel(r'displacement error  $\epsilon$'); axs.set_ylabel(r'$P(g|\epsilon)=1-P(e|\epsilon)$')
pn(axs,'(b)')
fig.tight_layout()
fig.savefig("Paper_Figures/readout_combined.pdf",bbox_inches="tight",transparent=True)
fig.savefig("Paper_Figures/readout_combined.png",dpi=300,bbox_inches="tight")
print("replotted: ideal@peak err=%.2e flag@peak err=%.2e Helstrom=%.2e flag_succ@peak=%.1f%%"
      %(np.min(np.abs(d['err_ideal'])),np.min(np.abs(d['err_flag'])),hel,100*d['succ_flag'][0]))
