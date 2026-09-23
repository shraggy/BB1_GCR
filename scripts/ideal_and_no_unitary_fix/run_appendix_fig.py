"""3-panel BB1(GCR) readout figure:
 (a) full-range readout P(+1): bare BB1, ideal non-unitary BB1(GCR), physical UNITARY BB1(GCR)
     (washes out), and the flagged-SUCCESS branch (herald 3 corrections + deterministic GCR target).
 (b) readout error P(-1) across the correctable region (log): bare BB1, single-GCR (finite-energy),
     and BB1(GCR)-success -- shows BB1(GCR)-success stays flat where single-GCR degrades.
 (c) oscillator back-action: reduced-oscillator fidelity vs the input across the correctable region.
     Bare BB1 mangles the state (0.99 -> 0.89); single-GCR and BB1(GCR)-success stay ~0.996.
No s-sweep / success-probability panel: s=1 (full GCR correction) is the only operating point, and
the 17% herald success is a single number quoted in the text.
"""
import time, numpy as np
from qutip import *
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
_rob=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts","Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
# match the paper style (NA-QSP_sims.ipynb cell 13): thick spines, inward ticks, cm mathtext
plt.rcParams.update({'font.family':'Roboto','mathtext.fontset':'cm',
                     'axes.labelsize':19,'xtick.labelsize':15,'ytick.labelsize':15,
                     'axes.linewidth':3,
                     'xtick.direction':'in','ytick.direction':'in',
                     'xtick.major.width':2,'ytick.major.width':2,
                     'xtick.major.size':4,'ytick.major.size':4,
                     'xtick.minor.width':1.5,'ytick.minor.width':1.5,
                     'xtick.minor.size':2.5,'ytick.minor.size':2.5,
                     'legend.fontsize':11})
def panel(a,lab,x=-0.14,y=1.03): a.text(x,y,lab,transform=a.transAxes,fontsize=18,fontweight='bold',va='bottom',ha='right',clip_on=False)
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi):       return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_ideal(th,phi): return tensor((th*ep/sq)*pOp,sigma(phi)).expm(method='dense')            # e^{+c p sigma_phi}, non-unitary
def Opy_unit(th,phi):  return tensor(-1j*(th*ep/sq)*pOp,sigma(phi+np.pi/2)).expm(method='dense') # unitary GCR (n=+z axis)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]; TARGET=BB1[-1]

def build(mode):
    U=None; n=np.array([0,0,1.0])
    for i,(th,phi) in enumerate(BB1):
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if   mode=='bare':    op=ox
        elif mode=='ideal':   op=ox*Opy_ideal(th,phi)
        elif mode=='phys':
            gam=np.cross(n,phiv); gam=gam/np.linalg.norm(gam)
            op=ox*tensor(-1j*(th*ep/sq)*pOp, gam[0]*SX+gam[1]*SY+gam[2]*SZ).expm(method='dense')
        elif mode=='flagged': op=ox*(Opy_ideal(th,phi) if i<3 else Opy_unit(th,phi))  # 3 heralds + det. target
        U=op if U is None else op*U
        n=rot_bloch(n,phiv,th)
    return U
def build_gcr():   # single-GCR finite-energy readout: the target pulse alone
    th,phi=TARGET; return Opx(th,phi)*Opy_unit(th,phi)

Uops={'bare':build('bare'),'ideal':build('ideal'),'phys':build('phys'),
      'flagged':build('flagged'),'gcr':build_gcr()}

# --- (a) full-range readout response ---
N=121; alph=np.linspace(-2*sq,2*sq,N); m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(U): return np.array([np.real(expect(ket2dm(py),(U*tensor(it,g)).unit().ptrace(1))) for it in inits])
P={k:sweep(Uops[k]) for k in ['bare','ideal','phys','flagged','gcr']}
for k in P: print(f"[{time.time()-t0:.0f}s] (a) {k}: P(-1)@0={1-P[k][int(np.argmin(np.abs(m)))]:.3e}")

# --- (b),(c) grid: <x>/sqrt(pi) in [0,1.5]. (b) shows log P(+1) zoomed near the odd peak at 1;
#     (c) shows back-action infidelity over [0,1]. ---
mr=np.linspace(0.0,1.5,31)
rinits=[(displace(Ncav,(mm*sq+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for mm in mr]
def pp_region(U): return np.array([np.real(expect(ket2dm(py),(U*tensor(it,g)).unit().ptrace(1))) for it in rinits])  # P(+1)
def back_region(U):
    F=[]
    for it in rinits:
        rho=(U*tensor(it,g)).unit().ptrace(0)
        mx=float(np.real(expect(xOp,rho))); mp=float(np.real(expect(pOp,rho)))
        xin=float(np.real(expect(xOp,it))); pin=float(np.real(expect(pOp,it)))
        D=displace(Ncav,-((mx-xin)+1j*(mp-pin))/np.sqrt(2))
        F.append(float(np.real((it.dag()*(D*rho*D.dag())*it)[0,0])))
    return np.array(F)
PP={k:pp_region(Uops[k]) for k in ['bare','gcr','flagged','ideal']}
BACK={k:back_region(Uops[k]) for k in ['bare','gcr','flagged','ideal']}
i04=int(np.argmin(np.abs(mr-0.4)))
print(f"[{time.time()-t0:.0f}s] (c) back-action infidelity @0.4: bare={1-BACK['bare'][i04]:.3e} gcr={1-BACK['gcr'][i04]:.3e} flagged={1-BACK['flagged'][i04]:.3e} ideal={1-BACK['ideal'][i04]:.3e}")
np.savez("Paper_Data/bb1gcr_fig.npz",m=m,bare=P['bare'],ideal=P['ideal'],phys=P['phys'],flagged=P['flagged'],gcr=P['gcr'],
         mr=mr,pp_bare=PP['bare'],pp_gcr=PP['gcr'],pp_flag=PP['flagged'],pp_ideal=PP['ideal'],
         back_bare=BACK['bare'],back_gcr=BACK['gcr'],back_flag=BACK['flagged'],back_ideal=BACK['ideal'])

def clip(v): return np.clip(v,1e-7,None)
# paper palette (match Fig 8d / readout_combined): bare BB1=royalblue, single-GCR=firebrick,
# flagged=purple(solid), ideal=purple(dashed), physical washout=grey
GREEN='0.55'; ORANGE='purple'; PURPLE='firebrick'; BLUE='royalblue'
fig=plt.figure(figsize=(16,4.8)); gs=GridSpec(1,3,width_ratios=[1.6,1.05,1.05],wspace=0.32)
ax=fig.add_subplot(gs[0]); axb=fig.add_subplot(gs[1]); axc=fig.add_subplot(gs[2])
RED='purple'
# (a) full-range response: bare, single-GCR, ideal (dashed red), physical washout, heralded
hp=ax.plot(m,P['phys'],color=GREEN,lw=2.0)[0]
hg=ax.plot(m,P['gcr'],color=PURPLE,lw=2.0,zorder=7)[0]
hf=ax.plot(m,P['flagged'],color=ORANGE,lw=2.0,zorder=8)[0]
hi=ax.plot(m,P['ideal'],color=RED,lw=2.2,ls='--',zorder=9)[0]
hb=ax.plot(m,P['bare'],color=BLUE,lw=3.0,zorder=10)[0]
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.35)
panel(ax,'(a)',x=-0.13); ax.axvspan(-0.5,0.5,color='0.93',zorder=0)
# (b) log P(+1) zoomed near the odd peak at <x>/sqrt(pi)=1 (P(+1) -> small; log-informative)
axb.plot(mr,clip(PP['bare']),'o-',color=BLUE,lw=2.2,ms=3.5)
axb.plot(mr,clip(PP['gcr']),'s-',color=PURPLE,lw=2.0,ms=3.5)
axb.plot(mr,clip(PP['flagged']),'^-',color=ORANGE,lw=2.0,ms=3.5)
axb.plot(mr,clip(PP['ideal']),'--',color=RED,lw=2.2)
axb.set_yscale('log'); axb.set_xlim(0.5,1.5); axb.set_ylim(1e-7,1.6)
axb.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axb.set_ylabel(r'$P(+1)$')
axb.grid(alpha=0.35,which='both'); panel(axb,'(b)')
# (c) back-action infidelity 1-F over [0,1], log scale
axc.plot(mr,clip(1-BACK['bare']),'o-',color=BLUE,lw=2.2,ms=4)
axc.plot(mr,clip(1-BACK['gcr']),'s-',color=PURPLE,lw=2.0,ms=4)
axc.plot(mr,clip(1-BACK['flagged']),'^-',color=ORANGE,lw=2.0,ms=4)
axc.plot(mr,clip(1-BACK['ideal']),'--',color=RED,lw=2.2)
axc.set_yscale('log'); axc.set_xlim(0,1)
axc.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axc.set_ylabel(r'back-action infidelity $1-F$')
axc.grid(alpha=0.35,which='both'); panel(axc,'(c)')
# one shared legend at the bottom (colors are consistent across panels)
fig.tight_layout(); fig.subplots_adjust(left=0.05,right=0.995,bottom=0.34)
fig.legend([hb,hg,hi,hp,hf],
           ['bare BB1','single-GCR','ideal BB1(GCR), non-unitary','physical BB1(GCR), unitary',
            'BB1(GCR) heralded'],
           loc='lower center',ncol=3,fontsize=19,frameon=True,columnspacing=1.6,handlelength=2.0)
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight",transparent=True)
fig.savefig("Paper_Figures/appendix_fig.png",dpi=300,bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved figure")
