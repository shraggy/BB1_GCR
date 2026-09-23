"""
Reproduces Fig. 3 of the manuscript (BB1(GCR) vs BB1) from NA-QSP_sims.ipynb,
cells `2200a1f0` and `8f78613c`, plus diagnostics that test the user's hypothesis
("each pulse's infidelity makes the next one bad").

The composite-pulse operators do NOT depend on alpha (only the input state does),
so we build the matrix exponentials ONCE and reuse them -- ~100x faster than the
notebook's in-loop construction while producing identical results.

Metrics vs displacement error <x>/(2|alpha|):
  sy   = P(+1) of BB1(GCR)  ;  sy1 = P(+1) of bare BB1   (target = square wave)
  sfid = back-action 1-F_H of BB1(GCR) ; sfid1 = bare BB1
"""
import time
import numpy as np
from qutip import *

t0 = time.time()
g = basis(2, 0)
e = basis(2, 1)
py = (g + 1j * e).unit()

Ncav = 140
aOp = destroy(Ncav)
aOp1 = aOp.dag()
i = np.sqrt(2)
xOp = (aOp + aOp1) / i
pOp = (-1j) * (aOp - aOp1) / i

def sigma(phi):
    return np.cos(phi) * sigmax() + np.sin(phi) * sigmay()
def sigma_xyz(theta, phi):
    return (np.cos(theta) * sigmaz() + np.sin(theta) * sigma(phi))
def rot_xy(theta, phi):
    return (-1j * theta / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
def vec_f(vec_i, rot):
    return rot * vec_i * rot.dag()

Delta = 0.34
r = -np.log(Delta)
N = 120
a = -np.sqrt(np.pi) / 2
alph = np.linspace(-2 * np.sqrt(np.pi), 2 * np.sqrt(np.pi), N)
theta = np.pi / 2
phi0 = 0
phi1 = np.arccos(-theta / (4 * np.pi))
ep = Delta ** 2
x = np.pi / theta
beta = (theta / 2 / (np.sqrt(np.pi) / 2))

# ---- precompute the 4 GCR stages (each = x-kick * p-precorrection), alpha-independent ----
# stage order applied: Op4 (theta), Op3 (2pi), Op2 (pi), Op1 (theta) -- reversed BB1
vecf = vec_f(sigma_xyz(0, 0), rot_xy(0, 0))
vec = 1j * sigma(phi1) * vecf
G4 = tensor(-1j * x * beta * xOp, sigma(phi1)).expm(method='dense') * tensor(-1j * x * beta * (ep * pOp), vec * vecf).expm(method='dense')

vecf = vec_f(vecf, rot_xy(2 * np.pi, phi1))
vec = 1j * sigma(3 * phi1) * vecf
G3 = tensor(-2j * np.sqrt(np.pi) * xOp, sigma(3 * phi1)).expm(method='dense') * tensor(-2j * np.sqrt(np.pi) * (ep * pOp), vec * vecf).expm(method='dense')

vecf = vec_f(vecf, rot_xy(4 * np.pi, 3 * phi1))
vec = 1j * sigma(phi1) * vecf
G2 = tensor(-1j * np.sqrt(np.pi) * xOp, sigma(phi1)).expm(method='dense') * tensor(-1j * np.sqrt(np.pi) * (ep * pOp), vec * vecf).expm(method='dense')

vecf = vec_f(vecf, rot_xy(2 * np.pi, phi1))
vec = 1j * sigma(phi0) * vecf
G1 = tensor(-1j * beta * xOp, sigma(phi0)).expm(method='dense') * tensor(-1j * beta * (ep * pOp), vec).expm(method='dense')
GCR_STAGES = [G4, G3, G2, G1]

# bare BB1 stages (note .dag() as in notebook)
B4 = tensor(1j * np.sqrt(np.pi) * xOp, sigma(phi1)).expm(method='dense').dag()
B3 = tensor(2j * np.sqrt(np.pi) * xOp, sigma(3 * phi1)).expm(method='dense').dag()
B2 = tensor(1j * np.sqrt(np.pi) * xOp, sigma(phi1)).expm(method='dense').dag()
B1 = tensor(1j * beta * xOp, sigma(phi0)).expm(method='dense').dag()
BB1_STAGES = [B4, B3, B2, B1]


def GCR_BB1(initial, return_stages=False):
    state = tensor(initial, g)
    stages = []
    for Gk in GCR_STAGES:
        state = (Gk * state).unit()
        stages.append(state)
    return (state, stages) if return_stages else state


def BB1(initial):
    state = tensor(initial, g)
    for Bk in BB1_STAGES:
        state = (Bk * state).unit()
    return state


print(f"[setup {time.time()-t0:.1f}s] running sweep over {N} points, Ncav={Ncav} ...")
sy, sy1, sfid, sfid1 = [], [], [], []
for k, alpha1 in enumerate(alph):
    alpha = alpha1 + a
    initial = (displace(Ncav, alpha / np.sqrt(2)) * (squeeze(Ncav, r) * basis(Ncav, 0)).unit()).unit()
    state = GCR_BB1(initial)
    sy.append(np.abs(expect(ket2dm(py), state.ptrace(1))))
    sfid.append(1 - fidelity(ket2dm(initial), state.ptrace(0)))
    state = BB1(initial)
    sy1.append(np.abs(expect(ket2dm(py), state.ptrace(1))))
    sfid1.append(1 - fidelity(ket2dm(initial), state.ptrace(0)))

sy = np.array(sy); sy1 = np.array(sy1); sfid = np.array(sfid); sfid1 = np.array(sfid1)
err = alph / np.sqrt(np.pi)
np.savez("Paper_Data/GCR_BB1_alpha.npz", sy)
np.savez("Paper_Data/BB1_alpha.npz", sy1)
np.savez("Paper_Data/GCR_BB1_alpha_fid.npz", sfid)
np.savez("Paper_Data/BB1_alpha_fid.npz", sfid1)
np.savez("Paper_Data/GCR_BB1_err.npz", err)

print(f"\n=== Fig. 3 summary (theta=pi/2, Delta=0.34), done in {time.time()-t0:.1f}s ===")
print(f"BB1(GCR) back-action 1-F_H : min={sfid.min():.2e} median={np.median(sfid):.2e} max={sfid.max():.2e}")
print(f"bare BB1 back-action 1-F_H : min={sfid1.min():.2e} median={np.median(sfid1):.2e} max={sfid1.max():.2e}")
print(f"BB1(GCR) / BB1 median infidelity ratio = {np.median(sfid)/np.median(sfid1):.0f}x worse")
# square-wave quality of P(+1): how clean is the response near integer m vs midpoints
print(f"P(+1) BB1(GCR): min={sy.min():.3f} max={sy.max():.3f}  (clean square wave swings 0<->1)")
print(f"P(+1) bare BB1: min={sy1.min():.3f} max={sy1.max():.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={'width_ratios': [2, 1, 1]})
ax[0].plot(err, sy, '--', color="firebrick", lw=3, label="BB1(GCR)")
ax[0].plot(err, sy1, color="cornflowerblue", lw=2, label="BB1")
ax[0].set_title("Readout response P(+1)"); ax[0].set_xlabel(r"$\langle x\rangle/(2|\alpha|)$")
ax[0].set_ylabel("P(+1)"); ax[0].legend(); ax[0].grid()
ax[1].plot(err, sy, '--', color="firebrick", lw=3); ax[1].plot(err, sy1, color="cornflowerblue", lw=2)
ax[1].set_yscale("log"); ax[1].set_xlim(0, 2); ax[1].set_title("P(+1) (log)")
ax[1].set_xlabel(r"$\langle x\rangle/(2|\alpha|)$"); ax[1].grid()
ax[2].plot(err, sfid, '--', color="firebrick", lw=3, label="BB1(GCR)")
ax[2].plot(err, sfid1, color="cornflowerblue", lw=2, label="BB1")
ax[2].set_yscale("log"); ax[2].set_title(r"back-action infidelity $1-F_H$")
ax[2].set_xlabel(r"$\langle x\rangle/(2|\alpha|)$"); ax[2].legend(); ax[2].grid()
fig.tight_layout()
plt.savefig("Paper_Figures/GCR_BB1.png", dpi=130, bbox_inches="tight")
print("saved Paper_Figures/GCR_BB1.png")
