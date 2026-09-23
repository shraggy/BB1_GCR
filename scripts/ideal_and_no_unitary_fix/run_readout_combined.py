"""Combined GKP-readout figure (for BOTH the paper appendix and the note).
Panel (1): readout error P(e|eps) vs displacement error, paper colors:
   Infinite GKP (goldenrod), GCR/finite (firebrick), BB1 (royalblue), GCR-BB1 (forestgreen),
   BB1(GCR) post-selected IDEAL (purple dashed), heralded BB1(GCR) (purple SOLID),
   Helstrom bound (black dotted).
Panel (2): herald success probability vs eps for the two heralded schemes
   (full-ideal all-4 ~7%, heralded herald-3 ~17%) -- the discard cost.
Ncav via argv[1] (default 100 for a fast sanity run; use 400 to match the paper).

KEY: the user's GCR_BB1 uses correction axis vec*vecf = i*sigma_phi (anti-Hermitian) -> the
NON-UNITARY ideal, made a state by per-step renormalization = the post-selected success branch.
"""
import sys, time, numpy as np
from qutip import *
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 100
t0=time.time()
g=basis(2,0); e=basis(2,1); py=(g+1j*e).unit()
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def sigma_xyz(th,phi): return np.cos(th)*sigmaz()+np.sin(th)*sigma(phi)
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm()
def vec_f(v,rot): return rot*v*rot.dag()
Delta=0.34; r=-np.log(Delta); theta=np.pi/2; phi0=0.0; phi1=np.arccos(-theta/(4*np.pi)); ep=Delta**2
x=np.pi/theta; beta=(theta/2/(np.sqrt(np.pi)/2))
def logical(mu,delta,Ncav):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+3
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
I2=qeye(2)
def K(coef,phi): return tensor(-1j*coef*xOp, sigma(phi)).expm()          # position kick (GCR_BB1 sign)
def Cideal(coef,phi): return tensor(coef*(ep*pOp), sigma(phi)).expm()     # non-unitary ideal e^{+c p sigma_phi}
def Cunit(coef): return tensor(-1j*coef*(ep*pOp), sigmay()).expm()        # deterministic GCR (axis sigma_y)
# --- build eps-independent operators (products) once ---
# BB1(GCR) ideal (all 4 corrections non-unitary): U = Op1 Op2 Op3 Op4, Op_k = K * Cideal
Op4=K(x*beta,phi1)*Cideal(x*beta,phi1)
Op3=K(2*np.sqrt(np.pi),3*phi1)*Cideal(2*np.sqrt(np.pi),3*phi1)
Op2=K(np.sqrt(np.pi),phi1)*Cideal(np.sqrt(np.pi),phi1)
Op1=K(beta,phi0)*Cideal(beta,phi0)
U_ideal      = Op1*Op2*Op3*Op4                 # all-4 non-unitary (post-selected ideal)
U_flag       = (K(beta,phi0)*Cunit(beta))*Op2*Op3*Op4   # herald 3, deterministic target
# deterministic references (their exact sequences), built as products:
U_BB1  = (tensor(1j*beta*xOp,sigma(phi0)).expm().dag()*tensor(1j*np.sqrt(np.pi)*xOp,sigma(phi1)).expm().dag()
          *tensor(2j*np.sqrt(np.pi)*xOp,sigma(3*phi1)).expm().dag()*tensor(1j*np.sqrt(np.pi)*xOp,sigma(phi1)).expm().dag())
U_inf  = tensor(1j*beta*xOp,sigma(phi0)).expm()                                   # single CD
U_gcr  = tensor(1j*beta*xOp,sigma(phi0)).expm()*tensor(1j*(beta*ep)*pOp,sigmay()).expm()   # finite-energy
U_gcrbb1 = (tensor(1j*np.sqrt(np.pi)*xOp,sigma(phi1)).expm()*tensor(2j*np.sqrt(np.pi)*xOp,sigma(3*phi1)).expm()
            *tensor(1j*np.sqrt(np.pi)*xOp,sigma(phi1)).expm()*tensor(1j*beta*xOp,sigma(phi0)).expm()
            *tensor(1j*(beta*ep/4)*pOp,sigmay()).expm())                          # prepended-CD GCR-BB1
# --- prepended-block variants (GCR-BB1 block A / block B), EXACT conventions from
# run_prepended_block_compare.py / run_mesolve_4.py: SIG=sigma, BB1 angles/phi1 as above,
# Cdet(c,nu)=exp(-i c pO SIG(nu)), kick_i=exp(-i (th_i/sqrt(pi)) xO SIG(ph_i)), BB1U=K3K2K1K0 (K0 first),
# Cblock(p)=C3 C2 C1 C0 (C0 first), block unitary = BB1U * Cblock(p).
sqpi=np.sqrt(np.pi)
BB1ang=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICKb=[tensor(-1j*(th/sqpi)*xOp,sigma(ph)).expm() for th,ph in BB1ang]
BB1Ublock=None
for _K in KICKb: BB1Ublock = _K if BB1Ublock is None else _K*BB1Ublock   # K0 first
def CdetB(c,nu): return tensor(-1j*c*pOp,sigma(nu)).expm()
def Cblock(p):
    Cb=None
    for k in range(4):
        C=CdetB(p[2*k],p[2*k+1]); Cb = C if Cb is None else C*Cb   # C0 first
    return Cb
_bp=np.load('Paper_Data/prepended_block_params.npz')
paramsA=_bp['variantA_params']; paramsB=_bp['variantB_params']
U_blockA = BB1Ublock*Cblock(paramsA)
U_blockB = BB1Ublock*Cblock(paramsB)
print(f"[{time.time()-t0:.0f}s] operators built (Ncav={Ncav})")
tgt=logical(0,Delta,Ncav)
# Helstrom bound from logical overlap
ov=abs((logical(0,Delta,Ncav).dag()*logical(1,Delta,Ncav))[0,0]); helstrom=0.5*(1-np.sqrt(1-ov**2))
print(f"[{time.time()-t0:.0f}s] logical overlap={ov:.3e}  Helstrom={helstrom:.3e}")
# --- eps grid over the correctable region (peak at eps_plot=0 -> boundary ~0.63) ---
NA=56; alph=np.linspace(-0.90,1.80,NA); eps=alph/np.sqrt(2)+np.sqrt(np.pi/2)/2  # eps in [~0, 1.9] (covers 3u)
inits=[(displace(Ncav,al/np.sqrt(2))*tgt).unit() for al in alph]
def Pp(U,it): return float(np.abs(expect(ket2dm(py),(U*tensor(it,g)).unit().ptrace(1))))
def err_curve(U,invert):
    return np.array([(1-Pp(U,it)) if invert else Pp(U,it) for it in inits])
# calibration (matches cell 15): infinite/GCR/GCR-BB1 use P(+1); BB1/BB1(GCR) use 1-P(+1)
ERR={}
ERR['inf']=err_curve(U_inf,False); ERR['gcr']=err_curve(U_gcr,False); ERR['gcrbb1']=err_curve(U_gcrbb1,False)
ERR['bb1']=err_curve(U_BB1,True); ERR['ideal']=err_curve(U_ideal,True); ERR['flag']=err_curve(U_flag,True)
ERR['blockA']=err_curve(U_blockA,True); ERR['blockB']=err_curve(U_blockB,True)   # KICKb uses the -1j (K()) sign
# convention, same as U_BB1's effective kicks (not U_gcrbb1's +1j raw kicks) -> use 1-P(+1), like 'bb1'/'ideal'/'flag'
for k in ERR: print(f"[{time.time()-t0:.0f}s] err {k}: peak(min)={np.min(np.abs(ERR[k])):.3e}")
# --- herald success probability vs eps ---
def lam_sub(U,nsub=40):
    M=U.full(); return np.linalg.svd(M[:, :2*nsub], compute_uv=False)[0]
# full raw operators (kicks unitary, corrections non-unitary): norm reduction sets success prob.
# all-4 non-unitary (ideal) reduces norm more -> lower success; 3-herald (flag) higher.
lam_ideal=lam_sub(U_ideal); lam_flag=lam_sub(U_flag)
def succ_curve(U,lam): return np.array([float((U*tensor(it,g)).norm()**2)/lam**2 for it in inits])
SUCC={'ideal':succ_curve(U_ideal,lam_ideal),'flag':succ_curve(U_flag,lam_flag)}
print(f"[{time.time()-t0:.0f}s] succ ideal@peak={SUCC['ideal'][0]:.3f} flag@peak={SUCC['flag'][0]:.3f}")
import os as _os
_npz_path="Paper_Data/readout_combined.npz"
_new=dict(eps=eps,helstrom=helstrom,**{f'err_{k}':ERR[k] for k in ERR},**{f'succ_{k}':SUCC[k] for k in SUCC})
if _os.path.exists(_npz_path):
    _d=dict(np.load(_npz_path)); _nold=len(_d)
    _d.update(_new); np.savez(_npz_path,**_d)
    print(f"[{time.time()-t0:.0f}s] merged & saved data ({_nold} keys -> {len(_d)} keys)")
else:
    np.savez(_npz_path,**_new)
    print(f"[{time.time()-t0:.0f}s] saved data ({len(_new)} keys, new file)")

# ------------------------------------------------------------------ plotting (paper style)
import os
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
GOLD='goldenrod'; FIRE='firebrick'; ROY='royalblue'; FOR='forestgreen'; PUR='purple'
MSG='mediumseagreen'; DCY='darkcyan'
def clip(v): return np.clip(np.abs(v),1e-8,None)
fig=plt.figure(figsize=(12,5)); gs=GridSpec(1,2,width_ratios=[1.35,1.0],wspace=0.28)
ax=fig.add_subplot(gs[0]); axs=fig.add_subplot(gs[1])
def pn(a,l): a.text(-0.12,1.02,l,transform=a.transAxes,fontsize=20,fontweight='bold',va='bottom')
# panel (a): readout error vs eps
ax.plot(eps,clip(ERR['inf']),color=GOLD,lw=2,label='Infinite GKP')
ax.plot(eps,clip(ERR['gcr']),color=FIRE,lw=2,label='GCR')
ax.plot(eps,clip(ERR['bb1']),color=ROY,lw=2,label='BB1')
ax.plot(eps,clip(ERR['gcrbb1']),color=FOR,lw=2,label='GCR-BB1')
ax.plot(eps,clip(ERR['blockA']),color=MSG,lw=2,label='GCR-BB1 (block A)')
ax.plot(eps,clip(ERR['blockB']),color=DCY,lw=2,label='GCR-BB1 (block B)')
ax.plot(eps,clip(ERR['ideal']),'--',color=PUR,lw=2.4,label='BB1(GCR) ideal')
ax.plot(eps,clip(ERR['flag']),'-',color=PUR,lw=2.4,label='BB1(GCR) heralded')
ax.axhline(helstrom,ls=':',color='k',lw=2,label='Helstrom bound')
ax.set_yscale('log'); ax.set_xlim(eps[0],eps[-1]); ax.grid(alpha=0.35,which='both')
ax.set_xlabel(r'displacement error  $\epsilon$'); ax.set_ylabel(r'readout error  $P(e|\epsilon)$')
pn(ax,'(a)'); ax.legend(fontsize=10.5,loc='lower right',frameon=True,framealpha=0.9)
# panel (b): herald success probability vs eps
axs.plot(eps,100*SUCC['flag'],'-',color=PUR,lw=2.4,label='heralded (herald 3)')
axs.plot(eps,100*SUCC['ideal'],'--',color=PUR,lw=2.4,label='full ideal (herald 4)')
axs.set_xlim(eps[0],eps[-1]); axs.set_ylim(0,100); axs.grid(alpha=0.35)
axs.set_xlabel(r'displacement error  $\epsilon$'); axs.set_ylabel('herald success (\\%)')
pn(axs,'(b)'); axs.legend(fontsize=11,loc='upper right',frameon=True)
axs.axhspan(0,100*np.max(SUCC['flag']),color='0.92',zorder=0)
fig.tight_layout()
fig.savefig("Paper_Figures/readout_combined.pdf",bbox_inches="tight",transparent=True)
fig.savefig("Paper_Figures/readout_combined.png",dpi=300,bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved figure")
