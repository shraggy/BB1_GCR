"""
Separate plots for BB1, SCROFULOUS, TYCKO (each: bare vs vec[physical GCR] vs
vec*vecf[ideal GCR]) for the modular readout, NET target = pi/2.

Generic GCR construction, faithfully generalized from the notebook GCR_BB1:
  vecf starts at sigma_z; for each pulse (theta_k about phi_k):
     vec  = 1j sigma(phi_k) vecf
     axis = vec*vecf (ideal, non-unitary)  OR  vec (physical)
     pulse = Opx(theta_k,phi_k) * Opy(theta_k,phi_k,axis)     [precorr applied first]
     update vecf <- rot_xy(2 theta_k, phi_k) vecf rot_xy(...)^dag
  x-kick coeff theta/(2|alpha|), precorr coeff theta*Delta^2/(2|alpha|), |alpha|=sqrt(pi)/2.

Validation (user): if the pulses are right, vec*vecf (ideal) should BEAT bare.
We also report ||U^dag U - I|| for the full composite per mode (vec*vecf is non-unitary).
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0 = time.time()
g = basis(2,0); py = (basis(2,0)+1j*basis(2,1)).unit()
Ncav = 140
aOp = destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm(method='dense')
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); d2r=np.pi/180
I2=tensor(qeye(Ncav),qeye(2))

def Opx(theta,phi): return tensor(-1j*(theta/sq)*xOp, sigma(phi)).expm(method='dense')
def Opy(theta,phi,axis): return tensor(-1j*(theta*ep/sq)*pOp, axis).expm(method='dense')

def build(seq, mode):
    ops=[]; vecf=sigmaz()
    for theta,phi in seq:
        ox=Opx(theta,phi)
        if mode=='bare':
            ops.append(ox)
        else:
            vec=1j*sigma(phi)*vecf
            axis=(vec*vecf) if mode=='vecvecf' else vec
            ops.append(ox*Opy(theta,phi,axis))
        R=rot_xy(2*theta,phi); vecf=R*vecf*R.dag()
    return ops

phi1=np.arccos(-1/8)
SEQS={
 'BB1':        [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)],
 'SCROFULOUS': [(115.2*d2r,62*d2r),(np.pi,280.6*d2r),(115.2*d2r,62*d2r)],   # net pi/2 (theta=90 row)
 'TYCKO':      [(385*d2r,0.0),(320*d2r,np.pi),(25*d2r,0.0)],                # net pi/2 (off-resonance)
}
MODES=['bare','vec','vecvecf']
COLORS={'bare':'cornflowerblue','vec':'seagreen','vecvecf':'firebrick'}
LABELS={'bare':'bare (no GCR)','vec':'vec (physical GCR)','vecvecf':'vec*vecf (ideal GCR)'}

built={(c,m):build(s,m) for c,s in SEQS.items() for m in MODES}
print(f"[setup {time.time()-t0:.0f}s] built {len(built)} (composite,mode) operators")

# unitarity of full composite per (composite,mode)
def Ufull(ops):
    U=ops[0]
    for o in ops[1:]: U=o*U
    return U
print("\n||U^dag U - I|| (full composite):")
for c in SEQS:
    for m in MODES:
        U=Ufull(built[(c,m)]); print(f"   {c:<11} {m:<8}: {(U.dag()*U-I2).norm():.2e}")

N=41
a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2; m=a1/sq
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
def run(ops,init):
    s=tensor(init,g)
    for U in ops: s=(U*s).unit()
    return np.real(expect(ket2dm(py),s.ptrace(1)))

P={}
for c in SEQS:
    for mm in MODES:
        P[(c,mm)]=np.array([run(built[(c,mm)],it) for it in inits])
    print(f"[{time.time()-t0:.0f}s] swept {c}")

# operating-point metric per composite: plateaus defined by ITS OWN vec*vecf curve
print("\n=== operating-point readout error (defined by each composite's vec*vecf plateaus) ===")
summary={}
for c in SEQS:
    pid=P[(c,'vecvecf')]; mask=np.abs(pid-np.round(pid))<0.07; tgt=np.round(pid)
    if mask.sum()==0: mask=np.ones_like(pid,bool)
    print(f"\n{c}  ({mask.sum()} operating pts):")
    for mm in MODES:
        err=np.mean(np.abs(P[(c,mm)]-tgt)[mask])
        summary[(c,mm)]=err
        print(f"    {LABELS[mm]:<22}: {err:.4f}")
    chk = summary[(c,'vecvecf')] < summary[(c,'bare')]
    print(f"    --> vec*vecf beats bare? {chk}  (validation: pulses doing the right thing)")

np.savez("Paper_Data/three.npz", m=m, **{f"{c}_{mm}":P[(c,mm)] for c in SEQS for mm in MODES})

# three separate plots
fig,ax=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for i,c in enumerate(SEQS):
    for mm in MODES:
        ls = '--' if mm=='vecvecf' else ('-' if mm=='bare' else '-.')
        lw = 2.5 if mm=='vecvecf' else 2
        ax[i].plot(m,P[(c,mm)],ls,color=COLORS[mm],lw=lw,
                   label=f"{LABELS[mm]} (op-err {summary[(c,mm)]:.3f})")
    ax[i].set_title(f"{c}  (net $\\theta=\\pi/2$)"); ax[i].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$")
    ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8,loc='lower center')
ax[0].set_ylabel("P(+1)")
fig.suptitle("Readout response per composite: bare vs vec (physical) vs vec*vecf (ideal)",fontsize=13)
fig.tight_layout(); fig.savefig("Paper_Figures/three_composites.png",dpi=130,bbox_inches="tight")
print("\nsaved Paper_Figures/three_composites.png")
