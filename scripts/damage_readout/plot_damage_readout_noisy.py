"""Damage-then-NOISY-readout figures for the note (companion to plot_damage_readout.py).

plot_damage_readout.py covers only the NOISELESS-readout condition (isolating the effect of
pre-readout input damage). This script covers the five READOUT-circuit noise conditions
('complete','tr_decay','tr_deph','osc_decay','osc_deph') so the reader can see how adding
readout-circuit noise on top of pre-readout oscillator damage affects both the readout error
P(e) and the post-read oscillator infidelity 1-F.

5x2 panel grid, matching the paired-plot aesthetic (run_paired_plots.py):
  rows    : the five noisy readout conditions
  columns : oscillator DECAY damage channel (left) | oscillator DEPHASING damage channel (right)
Each panel overlays the SEVEN readout protocols vs damage time (us), averaged over the
logical-0/1 codewords. Colors/labels reuse the canonical SCOL/SLAB mapping from
run_paired_plots.py for cross-figure consistency. Reads Paper_Data/damage_readout.npz
(err + infid keys); does not run any physics/compute.
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
CONDS=[('complete', 'all channels'),
       ('tr_decay', 'transmon decay'),
       ('tr_deph',  'transmon dephasing'),
       ('osc_decay','oscillator decay'),
       ('osc_deph', 'oscillator dephasing')]
# draw order: lowest-floor / most-overlapped schemes LAST so they sit on top and stay visible
ZORD={'BB1':1,'GCR-BB1':2,'GCR':3,'herald-RUS':4,'GCR-BB1-blockA':5,'GCR-BB1-blockB':6,'BB1(GCR)-QITE':7}

def avg(proto,ch,cond,metric):
    v0=np.asarray(d[f'{proto}|{ch}|{cond}|L0|{metric}'])
    v1=np.asarray(d[f'{proto}|{ch}|{cond}|L1|{metric}'])
    return 0.5*(v0+v1)

def make_fig(metric,ylab,ylim,fname):
    fig,ax=plt.subplots(5,2,figsize=(10.5,15),sharex=True)
    for i,(cond,clabel) in enumerate(CONDS):
        for j,(ch,ctitle) in enumerate(CH):
            a=ax[i,j]
            for sc in schemes:
                y=np.clip(avg(sc,ch,cond,metric),ylim[0]*0.9,None)
                ls=':' if sc=='herald-RUS' else '-'
                lw=2.4 if sc=='herald-RUS' else 1.8
                a.plot(t,y,ls,color=SCOL[sc],lw=lw,marker='o',ms=3.5,zorder=ZORD[sc],
                       label=SLAB[sc] if (i==0 and j==0) else None)
            a.set_yscale('log'); a.set_xlim(t[0],t[-1]); a.set_ylim(*ylim)
            a.grid(alpha=0.35)
            if i==0: a.set_title(ctitle,fontsize=13)
            if j==0: a.set_ylabel(f"{clabel}\n"+ylab,fontsize=12,fontweight='bold')
            if i==len(CONDS)-1: a.set_xlabel(r'idle damage time  $\tau$  ($\mu$s)')
    fig.legend(loc='lower center',ncol=4,fontsize=11,frameon=True,bbox_to_anchor=(0.5,-0.01))
    fig.tight_layout(rect=[0,0.035,1,1])
    fig.savefig(f"Paper_Figures/{fname}.pdf",bbox_inches="tight",transparent=True)
    fig.savefig(f"Paper_Figures/{fname}.png",dpi=150,bbox_inches="tight")
    print(f"wrote Paper_Figures/{fname}.pdf")
    plt.close(fig)

# ---- Figure 1: readout error P(e) ----
make_fig('err', r'$P(e)$', (3e-4,0.8), 'damage_readout_noisy')

# ---- Figure 2: post-read oscillator infidelity 1-F ----
# determine a sensible ylim from the actual data
_all=[]
for cond,_ in CONDS:
    for ch,_ in CH:
        for sc in schemes:
            _all.append(avg(sc,ch,cond,'infid'))
_all=np.concatenate([np.ravel(x) for x in _all])
gmin,gmax=float(np.min(_all)),float(np.max(_all))
print(f"[infid] global min={gmin:.3e}  max={gmax:.3e}")
# floor to a round decade at/just below 0.5*global_min
floor=10**np.floor(np.log10(0.5*gmin))
infid_ylim=(floor,1.2)
print(f"[infid] chosen ylim={infid_ylim}")

make_fig('infid', r'$1-F$', infid_ylim, 'damage_readout_noisy_infid')

# ---- numeric tables ----
print()
print("=== Figure 1: P(e) table ===")
for cond,_ in CONDS:
    for ch,_ in CH:
        for sc in schemes:
            e=avg(sc,ch,cond,'err')
            print(f"[P(e)][readcond={cond}][damage={ch}] {sc}: {e[0]:.3e} -> {e[-1]:.3e}")

print()
print("=== Figure 2: 1-F table ===")
for cond,_ in CONDS:
    for ch,_ in CH:
        for sc in schemes:
            e=avg(sc,ch,cond,'infid')
            print(f"[1-F][readcond={cond}][damage={ch}] {sc}: {e[0]:.3e} -> {e[-1]:.3e}")
