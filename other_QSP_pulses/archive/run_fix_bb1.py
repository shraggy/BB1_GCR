"""
Fix BB1 respecting the DETERMINISM constraint (user):
  GCR needs a predictable qubit trajectory to pick the precorrection axis. The
  position-sign-dependent (non-deterministic) readout rotation must happen ONLY in the
  final pulse -> the last (target pi/2) pulse must NOT be split; only the correction
  pulses (pi, 2pi, pi), where the qubit is on a known trajectory, may be split.

Also CHECK UNITARITY of each scheme: splitting tilts the tracked frame vecf, which can make
the physical 'vec' axis non-Hermitian (non-unitary) -- the same obstruction seen for
SCROFULOUS. A real 'fix' must (i) beat bare and (ii) stay unitary.

BB1 order = [ (pi,phi1), (2pi,3phi1), (pi,phi1), (pi/2,0) ]  ; idx 0,1,2 = corrections, 3 = target.
Schemes (physical vec GCR, K=8):
  bare, ideal(vecvecf)                         -- references
  all-split           : split 0,1,2,3          (breaks the last pulse -- user says wrong)
  corr-split          : split 0,1,2 ; target whole   <-- user's prescription
  target-split        : split only 3 ; corrections whole
  2pi-split           : split only idx1 (the 2pi) ; rest whole  (minimal fix)
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=120
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm(method='dense')
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,phi,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')

def build(seq,mode):
    ops=[]; vecf=sigmaz()
    for th,phi in seq:
        ox=Opx(th,phi)
        if mode=='bare': ops.append(ox)
        else:
            vec=1j*sigma(phi)*vecf; axis=(vec*vecf) if mode=='vecvecf' else vec
            ops.append(ox*Opy(th,phi,axis))
        R=rot_xy(2*th,phi); vecf=R*vecf*R.dag()
    return ops
def Ufull(ops):
    U=ops[0]
    for o in ops[1:]: U=o*U
    return U
def unit_err(ops): U=Ufull(ops); return (U.dag()*U-I2).norm()

phi1=np.arccos(-1/8)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
d2r=np.pi/180
SCROF=[(115.2*d2r,62*d2r),(np.pi,280.6*d2r),(115.2*d2r,62*d2r),(np.pi/2,0.0)]  # +pi/2 net? see note

# --- perpendicularity diagnostic: <zeta_k|sigma_phi_k|zeta_k> before each pulse (=0 => physical axis exists) ---
def perp_traj(seq,label):
    st=g; print(f"  {label}: <sigma_phi> before each pulse (0 = qubit perpendicular to pulse axis -> unitary correction exists)")
    for i,(th,phi) in enumerate(seq):
        val=np.real(expect(sigma(phi),st))
        print(f"     pulse {i} (theta={th/np.pi:.2f}pi, phi={phi/np.pi:.2f}pi): <sigma_phi>={val:+.3f}")
        st=((-1j*th/2*sigma(phi)).expm()*st).unit()
print("=== requirement 2 check: qubit perpendicular to pulse axis? ===")
perp_traj(BB1,"BB1")
perp_traj(SCROF[:3],"SCROFULOUS (3-pulse)")
def splitseq(split_idx,K):
    seq=[]
    for i,(th,phi) in enumerate(BB1):
        seq += [(th/K,phi)]*K if i in split_idx else [(th,phi)]
    return seq

N=31; a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def sweepP(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

K=8
Pideal=sweepP(build(BB1,'vecvecf'))
def op_err(P):
    mask=np.abs(Pideal-np.round(Pideal))<0.07; return np.mean(np.abs(P-np.round(Pideal))[mask])

schemes={
 'bare':            build(BB1,'bare'),
 'ideal vecvecf':   build(BB1,'vecvecf'),
 'all-split (K=8)':          build(splitseq({0,1,2,3},K),'vec'),
 'corr-split, target whole': build(splitseq({0,1,2},K),'vec'),   # user's prescription
 'target-split only':        build(splitseq({3},K),'vec'),
 '2pi-split only':           build(splitseq({1},K),'vec'),
}
print(f"[{time.time()-t0:.0f}s] scheme            op-err   ||U^dag U - I||  (unitary?)")
res={}
for name,ops in schemes.items():
    P=sweepP(ops); e=op_err(P); u=unit_err(ops); res[name]=(P,e,u)
    tag = "UNITARY" if u<1e-6 else "NON-UNITARY"
    print(f"  {name:<26} {e:.4f}   {u:.2e}   {tag}")

np.savez("Paper_Data/fix_bb1.npz", m=m, **{n.replace(' ','_').replace(',','').replace('(','').replace(')',''):res[n][0] for n in res})

# plot
fig,ax=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1.4,1]})
sty={'bare':('cornflowerblue','-'),'ideal vecvecf':('firebrick','--'),
     'all-split (K=8)':('gray','-'),'corr-split, target whole':('purple','-'),
     'target-split only':('orange',':'),'2pi-split only':('seagreen','-.')}
for n in schemes:
    c,ls=sty[n]; ax[0].plot(m,res[n][0],ls,color=c,lw=2.2,label=f"{n} ({res[n][1]:.3f})")
ax[0].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$"); ax[0].set_ylabel("P(+1)")
ax[0].set_title("BB1 readout: splitting schemes (physical vec GCR, K=8)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
names=list(schemes); errs=[res[n][1] for n in names]; us=[res[n][2] for n in names]
cols=['seagreen' if u<1e-6 else 'crimson' for u in us]
ax[1].barh(range(len(names)),errs,color=cols)
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names,fontsize=8); ax[1].invert_yaxis()
ax[1].axvline(res['bare'][1],ls='--',color='cornflowerblue',lw=1)
ax[1].set_xlabel("operating-point error"); ax[1].set_title("green=unitary(physical), red=non-unitary")
ax[1].grid(alpha=0.3,axis='x')
fig.tight_layout(); fig.savefig("Paper_Figures/fix_bb1.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/fix_bb1.png")
