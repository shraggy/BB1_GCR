"""
FLAGGED MODULAR READOUT for GKP  --  complete, self-contained reference implementation.
========================================================================================
This is how BB1(GCR) is turned into a *usable* modular readout.

WHY WE NEED IT.  BB1(GCR) only works with a NON-UNITARY "imaginary conditional
displacement"  M = exp(+ c p sigma_phi)  on each pulse (c = theta*Delta^2 / 2|alpha|).
M is not a physical gate, and every legal *unitary* substitute washes the readout out.

THE FIX (three-herald flagged readout).  Realize M *probabilistically* via an ancilla
herald on the THREE BB1 correction pulses {pi, 2pi, pi}, and leave the FINAL target pulse
{pi/2} a deterministic (unitary) GCR.  Because the target is deterministic:
    * EVERY shot returns a +/-1 bit (no post-selection of the outcome), and
    * the three heralds only FLAG the high-confidence shots.
On the ~17% of shots where all three heralds succeed the error is ~5.7e-4 (~10x below
bare BB1); the rest fall back to bare-level.  It is an end-of-line readout, so the
oscillator is discarded afterwards -- the modular (peak) information is all we use.

All numbers: Delta=0.34, |alpha|=sqrt(pi)/2, target theta_t=pi/2, peaks at <x> in sqrt(pi)*Z.
"""
import numpy as np
from qutip import *

# =====================================================================  CONSTANTS
Delta = 0.34               # GKP envelope (squeezing) parameter
ep    = Delta**2           # Delta^2 -- sets the correction strength c
sq    = np.sqrt(np.pi)     # sqrt(pi).  |alpha| = sqrt(pi)/2, so 2|alpha| = sqrt(pi) = sq
r     = -np.log(Delta)     # squeezing amplitude of an input GKP peak
a     = -sq/2              # centering offset: peak m sits at <x> = sqrt(pi)*(m - 1/2)
phi1  = np.arccos(-1/8)    # BB1 phase = arccos(-theta_t/4pi) with theta_t = pi/2

# BB1(GCR) pulses (theta, phi), applied first-to-last.
# The first THREE are the BB1 CORRECTIONS (heralded);  the LAST is the deterministic TARGET.
BB1 = [(np.pi,   phi1),     # 0 \
       (2*np.pi, 3*phi1),   # 1  |-- three corrections  -> heralded non-unitary M
       (np.pi,   phi1),     # 2 /
       (np.pi/2, 0.0)]      # 3  -- target -> deterministic GCR (NO ancilla, always applied)

# =====================================================================  OPERATORS
Ncav = 110
aOp  = destroy(Ncav)
xOp  = (aOp + aOp.dag())/np.sqrt(2)         # position quadrature x
pOp  = -1j*(aOp - aOp.dag())/np.sqrt(2)     # momentum quadrature p
SX, SY = sigmax(), sigmay()
g    = basis(2, 0)                          # qubit input |g>
pyk  = (basis(2, 0) + 1j*basis(2, 1)).unit()  # readout axis: P(-1) = 1 - <+i| rho_q |+i>
def sigma(phi):                             # equatorial Pauli  cos(phi) X + sin(phi) Y
    return np.cos(phi)*SX + np.sin(phi)*SY

# --- the two halves of each GCR unit, plus the deterministic-target variant -----------
# (A) POSITION KICK      e^{-i (theta/sq) x sigma_phi}          (theta/(2|alpha|) = theta/sq)
def Opx(th, phi):
    return tensor(-1j*(th/sq)*xOp, sigma(phi)).expm(method='dense')

# (B) IDEAL correction   M = e^{+ (theta*ep/sq) p sigma_phi}    (c = theta*Delta^2/2|alpha| = theta*ep/sq)
#     This is e^{-i c p (i sigma_phi)} -- the NON-UNITARY imaginary displacement.
#     "apply M then renormalise"  ==  the ancilla-herald SUCCESS branch (see Part 3).
def M_corr(th, phi, s=1.0):
    return tensor(s*(th*ep/sq)*pOp, sigma(phi)).expm(method='dense')

# (C) DETERMINISTIC GCR target: physical unitary correction, Hermitian axis sigma_{phi+pi/2}
def Opy_det(th, phi):
    return tensor(-1j*(th*ep/sq)*pOp, sigma(phi+np.pi/2)).expm(method='dense')

# =====================================================================  FLAGGED READOUT OP
# pulses 0,1,2 : Opx * M^s   (heralded non-unitary, success branch)
# pulse  3     : Opx * Opy_det   (deterministic GCR target)
def flagged_op(s=1.0):
    U = None
    for i, (th, phi) in enumerate(BB1):
        op = Opx(th, phi) * (M_corr(th, phi, s) if i < 3 else Opy_det(th, phi))
        U = op if U is None else op*U
    return U

def corr3_op(s=1.0):   # just the three non-unitary corrections -> sets herald success prob
    U = None
    for th, phi in BB1[:3]:
        op = Opx(th, phi) * M_corr(th, phi, s)
        U = op if U is None else op*U
    return U

# =====================================================================  INPUT PEAKS + READ
def peak_state(m):     # squeezed GKP peak centered at <x> = sqrt(pi)*(m - 1/2)
    al = m*sq
    return (displace(Ncav, (al+a)/np.sqrt(2)) * (squeeze(Ncav, r)*basis(Ncav, 0)).unit()).unit()

def read_Pminus(U, psi):   # apply readout U to |psi>|g>, renormalise (=herald success), P(-1)
    out = (U * tensor(psi, g)).unit()
    return 1 - float(np.real(expect(ket2dm(pyk), out.ptrace(1))))

def herald_success(s=1.0, sub=40):   # P_succ = ||Mcorr3 psi||^2 / lambda^2  (subspace-tailored)
    Mc = corr3_op(s); psi0 = peak_state(0.0)
    lam = np.linalg.svd(Mc.full()[:, :sub*2], compute_uv=False)[0]  # top singular value on n<=sub
    return float((Mc*tensor(psi0, g)).norm()**2) / lam**2

# =====================================================================  PART 1: the square wave
print("PART 1  flagged readout is a correct square wave (s=1):")
print(f"  {'m':>3} {'<x>':>8} {'P(-1)':>10}   bit  (want)")
Uf = flagged_op(1.0)
for m in [0.0, 1.0, 2.0, 3.0]:
    psi = peak_state(m); P = read_Pminus(Uf, psi)
    got = '+1' if P < 0.5 else '-1'; want = '+1' if int(np.floor(m)) % 2 == 0 else '-1'
    print(f"  {m:>3} {float(np.real(expect(xOp,psi))):>8.3f} {P:>10.3e}   {got}   ({want})")

# =====================================================================  PART 2: tradeoff vs s
print("\nPART 2  correction-strength sweep (data behind the tradeoff curve):")
print(f"  {'s':>5} {'flagged P(-1)@0':>16} {'herald success':>15}")
for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    e = read_Pminus(flagged_op(s), peak_state(0.0))
    print(f"  {s:>5} {e:>16.3e} {herald_success(s):>15.3e}")

# =====================================================================  PART 3: the herald, gate by gate
# One correction pulse (angle theta, axis phi, strength c=theta*ep/sq):
#   1. fresh ancilla A = |0>
#   2. W = exp(-i c p sigma_phi (x) sigma_y^A)   (system CD, additionally A-controlled via sigma_y)
#   3. Hadamard on A; measure A
#   4. HERALD: keep A=0 (success)  ->  applies K0 = (cos(cp) + sin(cp) sigma_phi)/sqrt2 ~ M
# The exact block-encoding success branch IS "apply M, renormalise" (used above); the simple
# W+H gadget reproduces M to FIRST ORDER in c. We quantify that gap here.
def cos_cp(c):
    U = ((1j*c)*pOp).expm(method='dense'); return 0.5*(U + U.dag())
def sin_cp(c):
    U = ((1j*c)*pOp).expm(method='dense'); return (U - U.dag())/(2j)
def K0(th, phi):                      # first-order gadget success Kraus
    c = th*ep/sq
    return (tensor(cos_cp(c), qeye(2)) + tensor(sin_cp(c), sigma(phi)))/np.sqrt(2)

print("\nPART 3  gadget (W,H,measure) vs exact M -- fidelity of the success state per pulse:")
psi0 = tensor(peak_state(0.0), g)
for th, phi in BB1[:3]:
    exact = (M_corr(th, phi)*psi0).unit()      # exact block-encoding success branch
    gadget = (K0(th, phi)*psi0).unit()         # first-order W+H gadget success branch
    F = float(np.abs(exact.overlap(gadget))**2)
    print(f"  theta={th/np.pi:>4.2f}pi  axis phi={phi:>5.2f}  c={th*ep/sq:>5.3f}   fidelity(gadget, exact) = {F:.5f}")
