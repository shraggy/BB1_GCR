"""
Alternatives to BB1(GCR) for the modular readout (all PHYSICAL / unitary), theta_t=pi/2.

Composite pulses rendered as position-controlled rotations (like the notebook):
  x-kick:        Opx(a,phi) = exp(-i (a/sqrt(pi)) x_hat sigma_phi)       [|alpha|=sqrt(pi)/2]
  GCR precorr:   Opy(a,phi) = exp(-i (a*Delta^2/sqrt(pi)) p_hat sigma_{phi+90}) (applied first)
A pulse is  Opx*Opy  in "gcr" mode, or just Opx in "bare" mode.

Sequences (area in deg, phase in deg):
  single      : (90,0)
  BB1         : (180,phi1)(360,3*phi1)(180,phi1)(90,0),  phi1=arccos(-1/8)=97.2 deg   <- has the 2pi pulse
  SCROFULOUS  : (115.2,62)(180,280.6)(115.2,62)          <- 3 pulses, max angle 180, = Tycko family
  TYCKO90(off): (385,0)(320,180)(25,0)                   <- OFF-RESONANCE corrector (wrong error class)
  hybrid      : BB1 with GCR on every pulse EXCEPT the 2pi one (which stays a bare rotation)

Metric: mean |P(+1) - nearest(0/1)| over the readout sweep (lower = better square wave).
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t0 = time.time()
g = basis(2, 0); py = (basis(2,0) + 1j*basis(2,1)).unit()
Ncav = 140
aOp = destroy(Ncav)
xOp = (aOp + aOp.dag())/np.sqrt(2); pOp = (-1j)*(aOp - aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax() + np.sin(phi)*sigmay()
Delta = 0.34; ep = Delta**2; r = -np.log(Delta); sq = np.sqrt(np.pi)
d2r = np.pi/180

def Opx(a, phi): return tensor(-1j*(a/sq)*xOp, sigma(phi)).expm(method='dense')
def Opy(a, phi): return tensor(-1j*(a*ep/sq)*pOp, sigma(phi+np.pi/2)).expm(method='dense')

phi1 = np.arccos(-1/8)
SEQS = {
  'single':      [(np.pi/2, 0.0)],
  'BB1':         [(np.pi, phi1), (2*np.pi, 3*phi1), (np.pi, phi1), (np.pi/2, 0.0)],
  # 5-pulse all-pi BB1: the 2pi pulse split into two pi pulses (Tycko-Pines/BB1-5 insight)
  'BB1split':    [(np.pi, phi1), (np.pi, 3*phi1), (np.pi, 3*phi1), (np.pi, phi1), (np.pi/2, 0.0)],
  'SCROFULOUS':  [(115.2*d2r, 62*d2r), (np.pi, 280.6*d2r), (115.2*d2r, 62*d2r)],
  'TYCKO90off':  [(385*d2r, 0.0), (320*d2r, np.pi), (25*d2r, 0.0)],
}
def is_2pi(a): return abs(a - 2*np.pi) < 1e-6

def build(seq, mode):
    """mode: 'bare', 'gcr', or 'hybrid' (gcr except the 2pi pulse)."""
    ops = []
    for a, phi in seq:
        if mode == 'bare':
            ops.append(Opx(a, phi))
        elif mode == 'gcr':
            ops.append(Opx(a, phi) * Opy(a, phi))
        elif mode == 'hybrid':
            ops.append(Opx(a, phi) if is_2pi(a) else Opx(a, phi) * Opy(a, phi))
    return ops

configs = {
  'single (bare)':          build(SEQS['single'], 'bare'),
  'BB1 (bare)':             build(SEQS['BB1'], 'bare'),
  'BB1 (GCR phys)':         build(SEQS['BB1'], 'gcr'),
  'hybrid: BB1 GCR, 2pi bare': build(SEQS['BB1'], 'hybrid'),
  'BB1 2pi->2xpi (GCR phys)': build(SEQS['BB1split'], 'gcr'),
  'SCROFULOUS (bare)':      build(SEQS['SCROFULOUS'], 'bare'),
  'SCROFULOUS (GCR phys)':  build(SEQS['SCROFULOUS'], 'gcr'),
  'TYCKO90off (bare)':      build(SEQS['TYCKO90off'], 'bare'),
  'TYCKO90off (GCR phys)':  build(SEQS['TYCKO90off'], 'gcr'),
}
print(f"[setup {time.time()-t0:.0f}s] built {len(configs)} configs")

N = 31
a1 = np.linspace(-2*sq, 2*sq, N); aoff = -sq/2
inits = [(displace(Ncav, (x+aoff)/np.sqrt(2)) * (squeeze(Ncav, r)*basis(Ncav,0)).unit()).unit() for x in a1]
m = a1/sq

def run(ops, init):
    s = tensor(init, g)
    for U in ops: s = (U*s).unit()
    return np.real(expect(ket2dm(py), s.ptrace(1)))

P = {}
for name, ops in configs.items():
    P[name] = np.array([run(ops, it) for it in inits])
    print(f"[{time.time()-t0:.0f}s] {name}")

def quality(p): return np.mean(np.abs(p - np.round(p)))
print("\n=========== readout quality: mean |P-nearest(0/1)| (lower=better) ===========")
order = sorted(configs, key=lambda k: quality(P[k]))
for k in order: print(f"   {k:<28}: {quality(P[k]):.3f}")
print("   (reference: BB1(GCR) vec*vecf non-physical ideal = 0.096 from earlier)")

np.savez("Paper_Data/alternatives.npz", m=m, **{k.replace(' ','_').replace(':','').replace('(','').replace(')',''): v for k,v in P.items()})

# --- plots ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios':[1.4,1]})
styles = {
  'single (bare)': ('0.6','-',1.5), 'BB1 (bare)': ('cornflowerblue','-',2),
  'BB1 (GCR phys)': ('navy','--',2), 'hybrid: BB1 GCR, 2pi bare': ('purple','-',2.5),
  'BB1 2pi->2xpi (GCR phys)': ('orange','-',2.5),
  'SCROFULOUS (bare)': ('firebrick','-',2), 'SCROFULOUS (GCR phys)': ('darkred','--',2.5),
  'TYCKO90off (bare)': ('green','-',1.5), 'TYCKO90off (GCR phys)': ('darkgreen',':',1.5),
}
for k in configs:
    c,ls,lw = styles[k]; ax[0].plot(m, P[k], ls, color=c, lw=lw, label=k)
ax[0].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$"); ax[0].set_ylabel("P(+1)")
ax[0].set_title(r"Modular readout response ($\theta_t=\pi/2$, $\Delta=0.34$)")
ax[0].legend(fontsize=7.5, ncol=2); ax[0].grid(alpha=0.3)
vals = [quality(P[k]) for k in order]
cols = [styles[k][0] for k in order]
ax[1].barh(range(len(order)), vals, color=cols)
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(order, fontsize=8)
ax[1].invert_yaxis(); ax[1].axvline(0.096, ls='--', color='k', lw=1, label='vec*vecf ideal (0.096)')
ax[1].set_xlabel("mean |P - nearest(0/1)|  (lower=better)"); ax[1].set_title("Readout quality")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='x')
fig.tight_layout(); fig.savefig("Paper_Figures/alternatives.png", dpi=130, bbox_inches="tight")
print("\nsaved Paper_Figures/alternatives.png")
