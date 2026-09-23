"""Damage-then-readout figure for the note (fig:damage).

2x2 panel grid, matching the paired-plot aesthetic (run_paired_plots.py):
  rows    : readout error P(e)  (top)  |  post-read oscillator infidelity 1-F (bottom)
  columns : oscillator DECAY (left)    |  oscillator DEPHASING (right)
Each panel overlays the SEVEN readout protocols vs damage time (us), NOISELESS readout
(isolating the effect of pre-readout input damage), averaged over the logical-0/1 codewords
(they are symmetric). Colors/labels reuse the canonical SCOL/SLAB mapping from run_paired_plots.py
for cross-figure consistency. Reads Paper_Data/damage_readout.npz (err + infid keys).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm','axes.labelsize':14,
    'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':2.2,
    'xtick.direction':'in','ytick.direction':'in','legend.fontsize':11,
    'xtick.major.width':1.6,'ytick.major.width':1.6,'xtick.major.size':4,'ytick.major.size':4})

d=np.load("Paper_Data/damage_readout.npz"); t=d['times']
kappa=float(d['kappa']); kappa_phi=float(d['kappa_phi'])
schemes=['GCR','BB1(GCR)-QITE','BB1','GCR-BB1','GCR-BB1-blockA','GCR-BB1-blockB','herald-RUS']
SCOL={'GCR':'firebrick','BB1(GCR)-QITE':'darkviolet','BB1':'royalblue','GCR-BB1':'forestgreen',
      'GCR-BB1-blockA':'mediumseagreen','GCR-BB1-blockB':'darkcyan','herald-RUS':'purple'}
SLAB={'GCR':'single GCR','BB1(GCR)-QITE':'BB1(GCR) det.','BB1':'bare BB1','GCR-BB1':'GCR-BB1 (1 pre-corr.)',
      'GCR-BB1-blockA':'GCR-BB1 (block A)','GCR-BB1-blockB':'GCR-BB1 (block B)','herald-RUS':'heralded'}
CH=[('osc_decay',  r'oscillator decay  ($T_1=1000\,\mu$s)'),
    ('osc_deph',   r'oscillator dephasing  ($\kappa_\phi^{-1}=5000\,\mu$s)')]
COND='noiseless'

def avg(proto,ch,metric):
    v0=np.asarray(d[f'{proto}|{ch}|{COND}|L0|{metric}'])
    v1=np.asarray(d[f'{proto}|{ch}|{COND}|L1|{metric}'])
    return 0.5*(v0+v1)

fig,ax=plt.subplots(2,2,figsize=(10.5,8.2),sharex=True)
ROWS=[('err',  r'readout error  $P(e)$',        (5e-6,0.7)),
      ('infid',r'post-read infidelity  $1-F$',  (1e-2,1.2))]
# draw order: lowest-floor / most-overlapped schemes LAST so they sit on top and stay visible
ZORD={'BB1':1,'GCR-BB1':2,'GCR':3,'herald-RUS':4,'GCR-BB1-blockA':5,'GCR-BB1-blockB':6,'BB1(GCR)-QITE':7}
for i,(metric,ylab,ylim) in enumerate(ROWS):
    for j,(ch,ctitle) in enumerate(CH):
        a=ax[i,j]
        for sc in schemes:
            y=np.clip(avg(sc,ch,metric),ylim[0]*0.9,None)
            ls=':' if sc=='herald-RUS' else '-'
            lw=2.4 if sc=='herald-RUS' else 1.8
            a.plot(t,y,ls,color=SCOL[sc],lw=lw,marker='o',ms=3.5,zorder=ZORD[sc],
                   label=SLAB[sc] if (i==0 and j==0) else None)
        a.set_yscale('log'); a.set_xlim(t[0],t[-1]); a.set_ylim(*ylim)
        a.grid(alpha=0.35)
        if i==0: a.set_title(ctitle,fontsize=13)
        if j==0: a.set_ylabel(ylab,fontsize=13)
        if i==1: a.set_xlabel(r'idle damage time  $\tau$  ($\mu$s)')
fig.legend(loc='lower center',ncol=4,fontsize=11,frameon=True,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.05,1,1])
fig.savefig("Paper_Figures/damage_readout.pdf",bbox_inches="tight",transparent=True)
fig.savefig("Paper_Figures/damage_readout.png",dpi=150,bbox_inches="tight")
print("wrote Paper_Figures/damage_readout.pdf")
# quick numeric echo for the caption/text
for ch,_ in CH:
    print(f"[{ch}] P(e) t=0..{t[-1]:.0f}:")
    for sc in schemes:
        e=avg(sc,ch,'err'); print(f"   {SLAB[sc]:22s} {e[0]:.4f} -> {e[-1]:.4f}")
