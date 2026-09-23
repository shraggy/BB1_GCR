"""Plot the mesolve results: readout error and oscillator impurity across the Fig-3 ranges
(panel a: eps in [0,u]; panel b: [2u,3u]) for the 4 cases x 5 noise conditions.
Produces Paper_Figures/mesolve_readout.pdf and mesolve_impurity.pdf."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm','axes.labelsize':16,
    'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':2,
    'xtick.direction':'in','ytick.direction':'in','xtick.major.width':1.5,'ytick.major.width':1.5,
    'legend.fontsize':11})
d=np.load("Paper_Data/mesolve_4.npz"); epsA=d['epsA']; epsB=d['epsB']; u=float(d['u'])
schemes=['GCR','BB1(GCR)-QITE','BB1','herald-RUS']
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
COL={'noiseless':'k','complete':'firebrick','tr_decay':'royalblue','tr_deph':'forestgreen','osc_decay':'darkorange'}
LS ={'noiseless':'--','complete':'-','tr_decay':'-','tr_deph':'-','osc_decay':'-'}
LAB={'noiseless':'noiseless','complete':'all channels','tr_decay':'transmon decay',
     'tr_deph':'transmon dephasing','osc_decay':'oscillator decay'}
def clip(v): return np.clip(np.abs(v),1e-8,None)
def make(metric, ylabel, fname, ylim):
    fig,ax=plt.subplots(len(schemes),2,figsize=(11,13))
    for i,sc in enumerate(schemes):
        for j,(eps,pan,lo,hi,ticks,tlab) in enumerate([
            (epsA,'a',epsA[0],u,[0,u],[r"$0$",r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"]),
            (epsB,'b',2*u,3*u,[2*u,3*u],[r"$\frac{\sqrt{\pi}}{\sqrt{2}}$",r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"])]):
            a=ax[i,j]
            for c in conds:
                a.plot(eps,clip(d[f'{sc}|{c}|{pan}|{metric}']),LS[c],color=COL[c],lw=2,label=LAB[c] if (i==0 and j==0) else None)
            a.set_yscale('log'); a.set_xlim(lo,hi); a.set_ylim(*ylim); a.grid(alpha=0.3,which='both')
            a.set_xticks(ticks); a.set_xticklabels(tlab)
            if j==0: a.set_ylabel(f"{sc}\n{ylabel}",fontsize=13)
            if i==len(schemes)-1: a.set_xlabel(r'displacement error $\epsilon$')
    fig.legend(loc='upper center',ncol=5,fontsize=12,frameon=True,bbox_to_anchor=(0.5,1.0))
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(f"Paper_Figures/{fname}.pdf",bbox_inches="tight")
    fig.savefig(f"Paper_Figures/{fname}.png",dpi=200,bbox_inches="tight")
    print("saved",fname)
make('q',   r'readout error $P(e|\epsilon)$', 'mesolve_readout',   (1e-6,1.0))
make('impur',r'oscillator impurity $1-\mathrm{Tr}\rho_o^2$','mesolve_impurity',(1e-4,1.0))
# print a compact operating-point (eps=0 / 2u) summary table
print("\n=== operating-point summary (eps=0 / logical-1 peak) ===")
for sc in schemes:
    print(f"\n{sc}:")
    for c in conds:
        print(f"  {c:10s} readout_err(a0)={d[f'{sc}|{c}|a|q'][0]:.3e}  impurity(a0)={d[f'{sc}|{c}|a|impur'][0]:.3e}")
