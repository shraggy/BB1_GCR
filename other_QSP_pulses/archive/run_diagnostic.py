"""
Diagnostic for the user's hypothesis:
  "BB1(GCR) is bad because the infidelity from each GCR pulse makes the next
   GCR pulse's input bad, so errors compound."

Three experiments (theta=pi/2, Delta=0.34):

 (1) INDIVIDUAL pulse: a single GCR rotation vs the same rotation done bare.
     Confirms each GCR pulse, in isolation, is *good* (the premise).

 (2) STAGE-BY-STAGE accumulation: instrument the 4-stage BB1(GCR) and report,
     after each stage, the oscillator back-action infidelity (1 - F to the ideal
     clean Gaussian input). If this grows monotonically, the input each GCR
     hands the next is progressively corrupted.

 (3) DECISIVE refresh test: re-run the composition but between stages restore the
     oscillator to a clean ideal Gaussian (optionally also reset the qubit to its
     ideal orientation), i.e. remove the corruption the previous pulse injected.
     If the refreshed composition is dramatically better than the true sequential
     one, the inter-stage corruption IS the mechanism -> hypothesis confirmed.
"""
import numpy as np
from qutip import *

g = basis(2, 0); e = basis(2, 1)
py = (g + 1j * e).unit()
Ncav = 160
aOp = destroy(Ncav); aOp1 = aOp.dag()
i = np.sqrt(2)
xOp = (aOp + aOp1) / i
pOp = (-1j) * (aOp - aOp1) / i

def sigma(phi): return np.cos(phi) * sigmax() + np.sin(phi) * sigmay()
def sigma_xyz(th, phi): return np.cos(th) * sigmaz() + np.sin(th) * sigma(phi)
def rot_xy(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
def vec_f(v, R): return R * v * R.dag()

Delta = 0.34; r = -np.log(Delta)
theta = np.pi / 2
phi0 = 0; phi1 = np.arccos(-theta / (4 * np.pi)); ep = Delta ** 2
x = np.pi / theta; beta = theta / 2 / (np.sqrt(np.pi) / 2)

# ---- 4 GCR stages (applied G4,G3,G2,G1) and their bare (x-kick only) counterparts ----
vecf = vec_f(sigma_xyz(0, 0), rot_xy(0, 0)); vec = 1j * sigma(phi1) * vecf
G4 = tensor(-1j * x * beta * xOp, sigma(phi1)).expm(method='dense') * tensor(-1j * x * beta * (ep * pOp), vec * vecf).expm(method='dense')
X4 = tensor(-1j * x * beta * xOp, sigma(phi1)).expm(method='dense')
vecf = vec_f(vecf, rot_xy(2 * np.pi, phi1)); vec = 1j * sigma(3 * phi1) * vecf
G3 = tensor(-2j * np.sqrt(np.pi) * xOp, sigma(3 * phi1)).expm(method='dense') * tensor(-2j * np.sqrt(np.pi) * (ep * pOp), vec * vecf).expm(method='dense')
X3 = tensor(-2j * np.sqrt(np.pi) * xOp, sigma(3 * phi1)).expm(method='dense')
vecf = vec_f(vecf, rot_xy(4 * np.pi, 3 * phi1)); vec = 1j * sigma(phi1) * vecf
G2 = tensor(-1j * np.sqrt(np.pi) * xOp, sigma(phi1)).expm(method='dense') * tensor(-1j * np.sqrt(np.pi) * (ep * pOp), vec * vecf).expm(method='dense')
X2 = tensor(-1j * np.sqrt(np.pi) * xOp, sigma(phi1)).expm(method='dense')
vecf = vec_f(vecf, rot_xy(2 * np.pi, phi1)); vec = 1j * sigma(phi0) * vecf
G1 = tensor(-1j * beta * xOp, sigma(phi0)).expm(method='dense') * tensor(-1j * beta * (ep * pOp), vec).expm(method='dense')
X1 = tensor(-1j * beta * xOp, sigma(phi0)).expm(method='dense')
GCR = [G4, G3, G2, G1]
BARE = [X4, X3, X2, X1]


def ideal_input(alpha):
    return (displace(Ncav, alpha / np.sqrt(2)) * (squeeze(Ncav, r) * basis(Ncav, 0)).unit()).unit()


def back_action_infid(osc_dm, initial_ket):
    return 1 - fidelity(ket2dm(initial_ket), osc_dm)


# ---------- (1) individual pulse ----------
print("=" * 72)
print("(1) INDIVIDUAL 2pi-pulse (the G3 stage): GCR vs bare, on a clean input")
print("=" * 72)
for m in [0.0, 1.0, 2.0]:
    alpha = m * 2 * (np.sqrt(np.pi) / 2)  # <x>/(2|alpha|) = m  (|alpha|=sqrt(pi)/2 here)
    init = ideal_input(alpha)
    st_gcr = (G3 * tensor(init, g)).unit()
    st_bare = (X3 * tensor(init, g)).unit()
    fb_gcr = back_action_infid(st_gcr.ptrace(0), init)
    fb_bare = back_action_infid(st_bare.ptrace(0), init)
    print(f"  m={m:>3}: 1-F_osc  GCR={fb_gcr:.2e}   bare={fb_bare:.2e}   "
          f"(GCR better by {fb_bare/max(fb_gcr,1e-12):.1f}x)")

# ---------- (2) stage-by-stage accumulation ----------
print("\n" + "=" * 72)
print("(2) STAGE-BY-STAGE oscillator back-action infidelity inside BB1(GCR)")
print("=" * 72)
for m in [0.3, 1.0]:
    alpha = m * 2 * (np.sqrt(np.pi) / 2)
    init = ideal_input(alpha)
    state = tensor(init, g)
    chain = []
    for Gk in GCR:
        state = (Gk * state).unit()
        chain.append(back_action_infid(state.ptrace(0), init))
    print(f"  m={m}:  after G4={chain[0]:.2e} -> G3={chain[1]:.2e} -> "
          f"G2={chain[2]:.2e} -> G1={chain[3]:.2e}")

# ---------- (3) decisive refresh test ----------
print("\n" + "=" * 72)
print("(3) REFRESH test: restore clean input between stages vs true sequential")
print("=" * 72)
print("   no-refresh  = true BB1(GCR);  refresh-osc = reset oscillator to clean")
print("   Gaussian between stages;  refresh-both = also reset qubit to ideal.\n")

def run_seq(init):
    st = tensor(init, g)
    for Gk in GCR: st = (Gk * st).unit()
    return st

def run_refresh(init, refresh_osc=True, refresh_qubit=False):
    # track ideal qubit trajectory with the bare (mean-field) rotations on |g>
    rho = ket2dm(tensor(init, g))
    qideal = tensor(init, g)
    for Gk, Xk in zip(GCR, BARE):
        rho = (Gk * rho * Gk.dag()).unit()
        qideal = (Xk * qideal).unit()
        rho_q = rho.ptrace(1)
        if refresh_qubit:
            rho_q = qideal.ptrace(1)
        if refresh_osc:
            rho = tensor(ket2dm(init), rho_q)   # clean Gaussian osc, drop correlations
    return rho

for m in [0.3, 1.0]:
    alpha = m * 2 * (np.sqrt(np.pi) / 2)
    init = ideal_input(alpha)
    s_seq = run_seq(init)
    r_osc = run_refresh(init, refresh_osc=True, refresh_qubit=False)
    r_both = run_refresh(init, refresh_osc=True, refresh_qubit=True)
    print(f"  m={m}:")
    print(f"     back-action 1-F_osc :  no-refresh={back_action_infid(s_seq.ptrace(0),init):.2e}"
          f"   refresh-osc={back_action_infid(r_osc.ptrace(0),init):.2e}"
          f"   refresh-both={back_action_infid(r_both.ptrace(0),init):.2e}")
    print(f"     P(+1) on qubit      :  no-refresh={np.real(expect(ket2dm(py),s_seq.ptrace(1))):.3f}"
          f"   refresh-osc={np.real(expect(ket2dm(py),r_osc.ptrace(1))):.3f}"
          f"   refresh-both={np.real(expect(ket2dm(py),r_both.ptrace(1))):.3f}")
print("\n(If refresh dramatically lowers infidelity, the inter-stage corruption"
      "\n IS the mechanism -> user's hypothesis confirmed.)")
