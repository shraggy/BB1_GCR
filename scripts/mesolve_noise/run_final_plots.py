"""Final figure generation for TASK A/B/C.
Produces:
  1. Updated Paper_Figures/mesolve_readout.pdf, mesolve_impurity.pdf
     -- add 5th case "GCR+sBs (repeated)" (Task B data) as a new row.
     -- on mesolve_impurity.pdf, overlay eps=0 sBs-corrected impurity (Task A "minimp") as star
        markers for the original 4 cases, for the 4 noisy conditions (no noiseless: impurity~0 there).
  2. NEW Paper_Figures/mesolve_paired_logical0.pdf (eps in [0,u]) and mesolve_paired_logical1.pdf
     (eps in [2u,3u]): top subplot = readout error, bottom = oscillator impurity, one line per case
     (all 6: GCR, BB1(GCR)-QITE, BB1, herald-RUS, GCR+sBs (repeated), GCR+sBs (heralded)), evaluated
     at the 'complete' (all-channels) noise condition -- the realistic composite case.
     Herald success probabilities (herald-RUS from run_herald_psucc.py; GCR+sBs (heralded) from
     Task C) are quoted as text annotations directly on both figures.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm','axes.labelsize':16,
    'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':2,
    'xtick.direction':'in','ytick.direction':'in','xtick.major.width':1.5,'ytick.major.width':1.5,
    'legend.fontsize':11})

d4  = np.load("Paper_Data/mesolve_4.npz")
dA  = np.load("Paper_Data/task_A_sbs.npz")
dB  = np.load("Paper_Data/task_B_gcrsbs.npz")
dC  = np.load("Paper_Data/task_C_gcrsbs_herald.npz")
dH  = np.load("Paper_Data/herald_psucc.npz")
u = float(d4['u']); epsA4 = d4['epsA']; epsB4 = d4['epsB']
epsAB = dB['epsA']; epsBB = dB['epsB']   # Task B/C share the same NE grid

schemes4 = ['GCR','BB1(GCR)-QITE','BB1','herald-RUS']
schemes5 = schemes4 + ['GCR+sBs (repeated)']
conds = ['noiseless','complete','tr_decay','tr_deph','osc_decay']
conds_noisy = ['complete','tr_decay','tr_deph','osc_decay']
COL={'noiseless':'k','complete':'firebrick','tr_decay':'royalblue','tr_deph':'forestgreen','osc_decay':'darkorange'}
LS ={'noiseless':'--','complete':'-','tr_decay':'-','tr_deph':'-','osc_decay':'-'}
LAB={'noiseless':'noiseless','complete':'all channels','tr_decay':'transmon decay',
     'tr_deph':'transmon dephasing','osc_decay':'oscillator decay'}
def clip(v): return np.clip(np.abs(v),1e-8,None)

def get_case_xyq(sc,cond,pan):
    """Return (eps, q) for readout error of case sc / cond / panel pan ('a' or 'b')."""
    if sc in schemes4:
        eps = epsA4 if pan=='a' else epsB4
        return eps, d4[f'{sc}|{cond}|{pan}|q']
    elif sc=='GCR+sBs (repeated)':
        eps = epsAB if pan=='a' else epsBB
        return eps, dB[f'GCR+sBs|{cond}|{pan}|q']
    elif sc=='GCR+sBs (heralded)':
        eps = epsAB if pan=='a' else epsBB
        return eps, dC[f'GCR+sBs-herald|{cond}|{pan}|q']

def get_case_ximp(sc,cond,pan):
    if sc in schemes4:
        eps = epsA4 if pan=='a' else epsB4
        return eps, d4[f'{sc}|{cond}|{pan}|impur']
    elif sc=='GCR+sBs (repeated)':
        eps = epsAB if pan=='a' else epsBB
        return eps, dB[f'GCR+sBs|{cond}|{pan}|impur']
    elif sc=='GCR+sBs (heralded)':
        eps = epsAB if pan=='a' else epsBB
        return eps, dC[f'GCR+sBs-herald|{cond}|{pan}|impur']

# ---------- 1. updated 5-row grids (mesolve_readout.pdf, mesolve_impurity.pdf) ----------
def make5(metric_fn, ylabel, fname, ylim, overlay_sbs=False):
    fig,ax=plt.subplots(len(schemes5),2,figsize=(11,15.5))
    for i,sc in enumerate(schemes5):
        for j,(pan,lo,hi,ticks,tlab) in enumerate([
            ('a',0,u,[0,u],[r"$0$",r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"]),
            ('b',2*u,3*u,[2*u,3*u],[r"$\frac{\sqrt{\pi}}{\sqrt{2}}$",r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"])]):
            a=ax[i,j]
            for c in conds:
                try:
                    eps,y = metric_fn(sc,c,pan)
                except KeyError:
                    continue
                a.plot(eps,clip(y),LS[c],color=COL[c],lw=2,label=LAB[c] if (i==0 and j==0) else None)
            if overlay_sbs and sc in schemes4 and pan=='a':
                # eps=0 sBs-corrected impurity (Task A "minimp"), 4 noisy conditions only
                for c in conds_noisy:
                    try:
                        mi = float(dA[f'{sc}|{c}|minimp'])
                    except KeyError:
                        continue
                    a.plot([0],[max(mi,1e-8)],marker='*',ms=13,mfc=COL[c],mec='k',mew=0.8,
                           linestyle='None', zorder=5,
                           label='after sBs (eps=0)' if (i==0 and j==0 and c=='complete') else None)
            a.set_yscale('log'); a.set_xlim(lo,hi); a.set_ylim(*ylim); a.grid(alpha=0.3,which='both')
            a.set_xticks(ticks); a.set_xticklabels(tlab)
            if j==0: a.set_ylabel(f"{sc}\n{ylabel}",fontsize=12)
            if i==len(schemes5)-1: a.set_xlabel(r'displacement error $\epsilon$')
    fig.legend(loc='upper center',ncol=3,fontsize=11,frameon=True,bbox_to_anchor=(0.5,1.0))
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f"Paper_Figures/{fname}.pdf",bbox_inches="tight")
    fig.savefig(f"Paper_Figures/{fname}.png",dpi=200,bbox_inches="tight")
    print("saved",fname)

make5(get_case_xyq,  r'readout error $P(e|\epsilon)$', 'mesolve_readout',   (1e-6,1.0))
make5(get_case_ximp, r'oscillator impurity $1-\mathrm{Tr}\rho_o^2$', 'mesolve_impurity', (1e-4,1.0), overlay_sbs=True)

# ---------- 2. paired plots (readout error + impurity, all cases, 'complete' condition) ----------
schemes6 = schemes4 + ['GCR+sBs (repeated)','GCR+sBs (heralded)']
COLc = {'GCR':'firebrick','BB1(GCR)-QITE':'royalblue','BB1':'forestgreen',
        'herald-RUS':'darkorange','GCR+sBs (repeated)':'purple','GCR+sBs (heralded)':'teal'}
cond_use='complete'

def herald_text(sc, pan):
    if sc=='herald-RUS':
        p = dH[f'herald-RUS|{cond_use}|{pan}|psucc']
        return f"herald-RUS success prob ({cond_use}): mean={p.mean():.3f}, range=[{p.min():.3f},{p.max():.3f}]"
    if sc=='GCR+sBs (heralded)':
        p = dC[f'GCR+sBs-herald|{cond_use}|{pan}|psucc']
        return f"GCR+sBs (heralded) cumulative success prob ({cond_use}): mean={p.mean():.3f}, range=[{p.min():.3f},{p.max():.3f}]"
    return None

def make_paired(pan, lo, hi, ticks, tlab, fname, title):
    fig,ax=plt.subplots(2,1,figsize=(8,8),sharex=True)
    texts=[]
    for sc in schemes6:
        try:
            eps,q = get_case_xyq(sc,cond_use,pan)
        except KeyError:
            continue
        ax[0].plot(eps,clip(q),'-o',ms=4,color=COLc[sc],lw=2,label=sc)
    for sc in schemes6:
        try:
            eps,im = get_case_ximp(sc,cond_use,pan)
        except KeyError:
            continue
        ax[1].plot(eps,clip(im),'-o',ms=4,color=COLc[sc],lw=2,label=sc)
        t = herald_text(sc,pan)
        if t: texts.append(t)
    for a in ax:
        a.set_yscale('log'); a.set_xlim(lo,hi); a.grid(alpha=0.3,which='both')
        a.set_xticks(ticks); a.set_xticklabels(tlab)
    ax[0].set_ylabel(r'readout error $P(e|\epsilon)$'); ax[0].set_ylim(1e-6,1.0)
    ax[1].set_ylabel(r'oscillator impurity $1-\mathrm{Tr}\rho_o^2$'); ax[1].set_ylim(1e-4,1.0)
    ax[1].set_xlabel(r'displacement error $\epsilon$')
    ax[0].set_title(f"{title}  (noise condition: {LAB[cond_use]})")
    ax[0].legend(loc='upper center',ncol=2,fontsize=9,frameon=True)
    if texts:
        ax[1].text(0.02,0.03,"\n".join(texts),transform=ax[1].transAxes,fontsize=8.5,
                    va='bottom',ha='left',bbox=dict(boxstyle='round',fc='white',ec='gray',alpha=0.9))
    fig.tight_layout()
    fig.savefig(f"Paper_Figures/{fname}.pdf",bbox_inches="tight")
    fig.savefig(f"Paper_Figures/{fname}.png",dpi=200,bbox_inches="tight")
    print("saved",fname)

make_paired('a', 0, u, [0,u], [r"$0$",r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"],
            'mesolve_paired_logical0', 'logical-0 operating range')
make_paired('b', 2*u, 3*u, [2*u,3*u], [r"$\frac{\sqrt{\pi}}{\sqrt{2}}$",r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"],
            'mesolve_paired_logical1', 'logical-1 operating range')

print("\n=== herald success-probability summary (for report) ===")
for sc in ['herald-RUS','GCR+sBs (heralded)']:
    for pan in ['a','b']:
        t = herald_text(sc,pan)
        print(f"{sc} panel-{pan}: {t}")
