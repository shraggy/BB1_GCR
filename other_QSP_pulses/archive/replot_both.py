"""Two separate figures (gray = notebook vec, green = correct axis), each with two log panels:
P(-1) near x=0 and P(+1) near x=sqrt(pi) (m=1), showing that construction's no-fix and split-fix
against bare and ideal. Reuses Paper_Data/fixes_both.npz (no recompute)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("Paper_Data/fixes_both.npz"); m=d["m"]
def clip(v): return np.clip(v,1e-7,None)

def make(tag, nofix_key, fix_key, color, fname, title):
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    curves=[('bare','tab:blue','-',2.0,'bare BB1'),
            ('ideal','firebrick','--',1.8,'ideal (non-unitary)'),
            (nofix_key,color,'-',2.2,f'{tag} no fix'),
            (fix_key,color,':',2.4,f'{tag} + split fix (K12)')]
    for k,c,ls,lw,lab in curves:
        ax[0].plot(m,clip(1-d[k]),ls,color=c,lw=lw,label=lab)
        ax[1].plot(m,clip(d[k]),ls,color=c,lw=lw,label=lab)
    ax[0].set_yscale('log'); ax[0].set_xlim(-0.7,0.7); ax[0].set_ylim(1e-5,1)
    ax[0].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[0].set_ylabel(r'$P(-1)$')
    ax[0].set_title(r'$P(-1)$ near $\langle x\rangle=0$'); ax[0].grid(alpha=0.3)
    ax[1].set_yscale('log'); ax[1].set_xlim(0.3,1.7); ax[1].set_ylim(1e-5,1)
    ax[1].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[1].set_ylabel(r'$P(+1)$')
    ax[1].set_title(r'$P(+1)$ near $\langle x\rangle=\sqrt{\pi}$'); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=9,loc='lower center')
    fig.suptitle(title,fontsize=13); fig.tight_layout()
    fig.savefig(fname,dpi=130,bbox_inches="tight"); print("saved",fname)

make('correct-axis','green_nofix','green_fix','#2ca02c',
     "Paper_Figures/fixes_green.png",
     r'Correct-axis ($\hat n\!\times\!\hat\phi$) BB1(GCR): no fix vs split fix, at the operating points')
make('notebook-vec','gray_nofix','gray_fix','0.35',
     "Paper_Figures/fixes_gray.png",
     r'Notebook-vec BB1(GCR): no fix vs split fix, at the operating points')
i0=int(np.argmin(np.abs(m))); i1=int(np.argmin(np.abs(m-1)))
print("P(-1)@x=0 :",{k:round(float(1-d[k][i0]),5) for k in ['bare','ideal','green_nofix','green_fix','gray_nofix','gray_fix']})
print("P(+1)@x=rt(pi):",{k:round(float(d[k][i1]),5) for k in ['bare','ideal','green_nofix','green_fix','gray_nofix','gray_fix']})
