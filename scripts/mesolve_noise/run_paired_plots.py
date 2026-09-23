"""Two metric-grouped figures (rows = noise condition, columns = logical-0 | logical-1 windows,
all protocols overlaid):
  * mesolve_paired_readout.pdf     -- readout error P(e|eps), from mesolve_4.npz (Ncav=400 mesolve).
  * mesolve_paired_infidelity.pdf  -- oscillator infidelity 1-F AFTER the single fixed corrective
        displacement (author's recipe), from corr_disp_fidelity.npz (key corr_fid = qutip fidelity F).
The infidelity uses the SAME fixed-beta_corr method as the note's fidelity table, not the per-state
mean removal in mesolve_4's 'ocorr' (which would disagree with the table)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm','axes.labelsize':14,
    'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':2.2,
    'xtick.direction':'in','ytick.direction':'in','legend.fontsize':11,
    'xtick.major.width':1.6,'ytick.major.width':1.6,'xtick.major.size':4,'ytick.major.size':4})
d=np.load("Paper_Data/mesolve_4.npz"); epsA=d['epsA']; epsB=d['epsB']; u=float(d['u'])
dc=np.load("Paper_Data/corr_disp_fidelity.npz")   # fixed-beta_corr fidelities (author's recipe)
schemes=['GCR','BB1(GCR)-QITE','BB1','GCR-BB1','GCR-BB1-blockA','GCR-BB1-blockB','herald-RUS']
CORRKEY={'GCR':'GCR','BB1(GCR)-QITE':'BB1(GCR)','BB1':'BB1','GCR-BB1':'GCR-BB1','GCR-BB1-blockA':'GCR-BB1-blockA','GCR-BB1-blockB':'GCR-BB1-blockB','herald-RUS':'heralded'}  # -> corr_disp scheme keys
SCOL={'GCR':'firebrick','BB1(GCR)-QITE':'darkviolet','BB1':'royalblue','GCR-BB1':'forestgreen','GCR-BB1-blockA':'mediumseagreen','GCR-BB1-blockB':'darkcyan','herald-RUS':'purple'}
SLAB={'GCR':'single GCR','BB1(GCR)-QITE':'BB1(GCR) det.','BB1':'bare BB1','GCR-BB1':'GCR-BB1 (1 pre-corr.)','GCR-BB1-blockA':'GCR-BB1 (block A)','GCR-BB1-blockB':'GCR-BB1 (block B)','herald-RUS':'heralded'}
conds=[('noiseless','noiseless'),('complete','all channels'),('tr_decay','transmon decay'),
       ('tr_deph','transmon dephasing'),('osc_decay','oscillator decay')]
def clip(v): return np.clip(np.abs(v),1e-8,None)
# columns = the two logical windows (logical-0 | logical-1); rows = noise conditions; all protocols.
COLS=[('a',epsA,0,u,[0,u],[r"$0$",r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"],r'logical-0  $\epsilon\in[0,u]$'),
      ('b',epsB,2*u,3*u,[2*u,3*u],[r"$\frac{\sqrt{\pi}}{\sqrt{2}}$",r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"],r'logical-1  $\epsilon\in[2u,3u]$')]
def make_metric(source, ylim, fname, title):
    fig,ax=plt.subplots(len(conds),2,figsize=(11.5,14))
    for i,(c,cname) in enumerate(conds):
        for j,(pan,eps,lo,hi,ticks,tlab,colname) in enumerate(COLS):
            a=ax[i,j]
            for sc in schemes:
                if source=='readout':
                    val=d[f'{sc}|{c}|{pan}|q']                       # readout error
                else:
                    val=1.0-dc[f'{CORRKEY[sc]}|{c}|{pan}|corr_fid']  # infidelity 1-F, fixed beta_corr
                ls=':' if sc=='herald-RUS' else '-'
                lw=2.4 if sc=='herald-RUS' else 1.8
                a.plot(eps,clip(val),ls,color=SCOL[sc],lw=lw,
                       label=SLAB[sc] if (i==0 and j==0) else None)
            a.set_yscale('log'); a.set_xlim(lo,hi); a.set_ylim(*ylim); a.grid(alpha=0.35)
            a.set_xticks(ticks); a.set_xticklabels(tlab)
            if j==0: a.set_ylabel(f"{cname}",fontsize=13,fontweight='bold')
            if i==0: a.set_title(colname,fontsize=13)
            if i==len(conds)-1: a.set_xlabel(r'displacement error $\epsilon$')
    fig.suptitle(title,fontsize=14,y=0.996)
    fig.legend(loc='lower center',ncol=4,fontsize=11,frameon=True,bbox_to_anchor=(0.5,-0.005))
    fig.tight_layout(rect=[0,0.02,1,0.98])
    fig.savefig(f"Paper_Figures/{fname}.pdf",bbox_inches="tight",transparent=True); fig.savefig(f"Paper_Figures/{fname}.png",dpi=150,bbox_inches="tight")
    print("wrote",fname)
make_metric('readout',   (1e-6,1),'mesolve_paired_readout',
            r'Readout error $P(e|\epsilon)$: each row a noise condition, columns logical-0/1, all protocols')
make_metric('infidelity',(1e-3,1),'mesolve_paired_infidelity',
            r'Oscillator infidelity $1-F$ (fixed corrective displacement): each row a noise condition, columns logical-0/1, all protocols')
