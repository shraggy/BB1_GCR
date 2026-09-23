"""
Three checks requested by the user, with plots.

PART A (composition, claims 1 & 2):
  BB1(GCR) with vec*vecf [non-unitary]  vs  BB1(GCR) with vec [unitary]  vs  bare BB1.
  -> vec*vecf is BETTER than BB1; vec is WORSE than BB1.

PART B (individual angles, claim 3):
  The standalone GCR(theta) primitive (2 gates, cell 697ba552) vs the standalone
  BB1(theta) sequence (4 gates, cell 26f0d1a7) for each angle theta in {pi/2, pi, 2pi}
  -> GCR beats BB1 at every angle.  (faithful port of the two cells "above BB1(GCR)")
"""
import time
import numpy as np
from qutip import *
from qutip.qip.operations import rx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
g = basis(2, 0); e = basis(2, 1)
py = (g + 1j * e).unit()

def sigma(phi): return np.cos(phi) * sigmax() + np.sin(phi) * sigmay()
def sigma_xyz(th, phi): return np.cos(th) * sigmaz() + np.sin(th) * sigma(phi)
def vec_f(v, R): return R * v * R.dag()

# =====================================================================
# PART A : composition  (Fig.3 units: i=sqrt(2))
# =====================================================================
NcavA = 140
aOp = destroy(NcavA); aOp1 = aOp.dag()
iA = np.sqrt(2)
xOpA = (aOp + aOp1) / iA
pOpA = (-1j) * (aOp - aOp1) / iA

def rot_xyA(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')

Delta = 0.34; r = -np.log(Delta)
thetaA = np.pi / 2
phi0 = 0; phi1 = np.arccos(-thetaA / (4 * np.pi)); ep = Delta ** 2
xfac = np.pi / thetaA; beta = thetaA / 2 / (np.sqrt(np.pi) / 2)

def build_stages(mode):
    A = (lambda v, vf: v * vf) if mode == 'vecvecf' else (lambda v, vf: v)
    out = []
    vf = vec_f(sigma_xyz(0, 0), rot_xyA(0, 0)); v = 1j * sigma(phi1) * vf
    out.append(tensor(-1j*xfac*beta*xOpA, sigma(phi1)).expm(method='dense') * tensor(-1j*xfac*beta*(ep*pOpA), A(v, vf)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(2*np.pi, phi1)); v = 1j*sigma(3*phi1)*vf
    out.append(tensor(-2j*np.sqrt(np.pi)*xOpA, sigma(3*phi1)).expm(method='dense') * tensor(-2j*np.sqrt(np.pi)*(ep*pOpA), A(v, vf)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(4*np.pi, 3*phi1)); v = 1j*sigma(phi1)*vf
    out.append(tensor(-1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense') * tensor(-1j*np.sqrt(np.pi)*(ep*pOpA), A(v, vf)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(2*np.pi, phi1)); v = 1j*sigma(phi0)*vf
    out.append(tensor(-1j*beta*xOpA, sigma(phi0)).expm(method='dense') * tensor(-1j*beta*(ep*pOpA), A(v, vf)).expm(method='dense'))
    return out

ST_nonU = build_stages('vecvecf')
ST_U = build_stages('vec')
B = [tensor(1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense').dag(),
     tensor(2j*np.sqrt(np.pi)*xOpA, sigma(3*phi1)).expm(method='dense').dag(),
     tensor(1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense').dag(),
     tensor(1j*beta*xOpA, sigma(phi0)).expm(method='dense').dag()]

def run_comp(stages, init):
    s = tensor(init, g)
    for U in stages:
        s = (U * s).unit()
    return s

print(f"[A setup {time.time()-t0:.0f}s] sweeping composition ...")
NA = 49
alphA = np.linspace(-2*np.sqrt(np.pi), 2*np.sqrt(np.pi), NA)
aoff = -np.sqrt(np.pi) / 2
mA, P_bb1, P_non, P_uni = [], [], [], []
F_bb1, F_non, F_uni = [], [], []
for a1 in alphA:
    alpha = a1 + aoff
    init = (displace(NcavA, alpha/np.sqrt(2)) * (squeeze(NcavA, r) * basis(NcavA, 0)).unit()).unit()
    s_b = run_comp(B, init); s_n = run_comp(ST_nonU, init); s_u = run_comp(ST_U, init)
    mA.append(a1/np.sqrt(np.pi))
    P_bb1.append(np.real(expect(ket2dm(py), s_b.ptrace(1))))
    P_non.append(np.real(expect(ket2dm(py), s_n.ptrace(1))))
    P_uni.append(np.real(expect(ket2dm(py), s_u.ptrace(1))))
    F_bb1.append(1 - fidelity(ket2dm(init), s_b.ptrace(0)))
    F_non.append(1 - fidelity(ket2dm(init), s_n.ptrace(0)))
    F_uni.append(1 - fidelity(ket2dm(init), s_u.ptrace(0)))
mA = np.array(mA)
np.savez("Paper_Data/checks_composition.npz", m=mA, P_bb1=P_bb1, P_non=P_non, P_uni=P_uni,
         F_bb1=F_bb1, F_non=F_non, F_uni=F_uni)
print(f"[A done {time.time()-t0:.0f}s]")

# =====================================================================
# PART B : individual-angle GCR vs BB1 primitives  (cells above BB1(GCR), i=2 units)
# =====================================================================
NcavB = 140
aOpB = destroy(NcavB); aOpB1 = aOpB.dag()
iB = 2
xOpB = (aOpB + aOpB1) / iB
pOpB = (-1j) * (aOpB - aOpB1) / iB
def rot_xyB(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')

alpha_list = np.arange(40, 130, 8) / 20.0  # 2.0 .. 6.05

def gcr_primitive(theta):
    """cell 697ba552: 2-gate GCR. tensor(osc, qubit)."""
    qub = rx(theta) * g
    sf = []
    for alpha in alpha_list:
        chi = -theta / 2 / alpha
        Opx = tensor(1j*chi*xOpB, sigmax()).expm(method='dense')
        Opy = tensor(1j*chi*pOpB, sigmay()).expm(method='dense')
        state = (Opx * Opy * tensor(coherent(NcavB, alpha), g)).unit()
        state2 = tensor(coherent(NcavB, alpha), qub).unit()
        state3 = (tensor(qeye(NcavB), ket2dm(qub)) * state).unit()
        sf.append(1 - np.abs(state3.overlap(state2))**2)
    return np.array(sf)

def bb1_primitive(theta):
    """cell 26f0d1a7 with ep=0: pure 4-gate BB1. tensor(qubit, osc)."""
    phi0 = 0; phi1 = np.arccos(-theta / (4 * np.pi)); ep = 0
    qub = rx(-theta) * g
    sf = []
    for alpha in alpha_list:
        initial = coherent(NcavB, alpha)
        xexp = np.abs(expect(xOpB, initial)); bet = theta / xexp
        state = tensor(g, initial)
        vf = vec_f(sigma_xyz(0, 0), rot_xyB(0, 0)); v = 1j*sigma(phi0)*vf
        Op1 = tensor(sigma(phi0), 1j*bet/2*xOpB).expm(method='dense') * tensor(v, 1j*bet/2*(ep*pOpB)).expm(method='dense')
        state = (Op1 * state).unit()
        vf = vec_f(vf, rot_xyB(np.pi/2, phi0)); v = 1j*sigma(phi1)*vf
        Op2 = tensor(sigma(phi1), 1j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 1j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense')
        state = (Op2 * state).unit()
        vf = vec_f(vf, rot_xyB(2*np.pi, phi1)); v = 1j*sigma(3*phi1)*vf
        Op3 = tensor(sigma(3*phi1), 2j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 2j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense')
        state = (Op3 * state).unit()
        vf = vec_f(vf, rot_xyB(4*np.pi, 3*phi1)); v = -1j*sigma(phi1)*vf
        Op4 = tensor(sigma(phi1), 1j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 1j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense')
        state = (Op4 * state).unit()
        state2 = tensor(qub, coherent(NcavB, alpha))
        state3 = (tensor(ket2dm(qub), qeye(NcavB)) * state).unit()
        sf.append(1 - np.abs(state3.overlap(state2))**2)
    return np.array(sf)

angles = [(np.pi/2, r"$\pi/2$"), (np.pi, r"$\pi$"), (2*np.pi, r"$2\pi$")]
indiv = {}
for th, lab in angles:
    print(f"[B angle {lab} {time.time()-t0:.0f}s] ...")
    indiv[lab] = (gcr_primitive(th), bb1_primitive(th))
np.savez("Paper_Data/checks_individual.npz", alpha=alpha_list,
         **{f"gcr_{k}": v[0] for k, v in indiv.items()},
         **{f"bb1_{k}": v[1] for k, v in indiv.items()})
print(f"[B done {time.time()-t0:.0f}s]")

# =====================================================================
# PLOTS
# =====================================================================
# Figure 1 : composition
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(mA, P_non, '--', color="firebrick", lw=3, label="BB1(GCR) vec*vecf [non-unitary]")
ax[0].plot(mA, P_bb1, color="cornflowerblue", lw=2, label="bare BB1 [unitary]")
ax[0].plot(mA, P_uni, '-', color="seagreen", lw=2, label="BB1(GCR) vec [unitary, physical]")
ax[0].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$"); ax[0].set_ylabel("P(+1)")
ax[0].set_title("Readout response"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(mA, F_non, '--', color="firebrick", lw=3, label="BB1(GCR) vec*vecf")
ax[1].plot(mA, F_bb1, color="cornflowerblue", lw=2, label="bare BB1")
ax[1].plot(mA, F_uni, '-', color="seagreen", lw=2, label="BB1(GCR) vec")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$")
ax[1].set_ylabel(r"back-action $1-F_H$"); ax[1].set_title("Back-action infidelity")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Claims 1 & 2: vec*vecf (non-unitary) beats BB1; vec (unitary) is worse than BB1", fontsize=12)
fig.tight_layout()
fig.savefig("Paper_Figures/check_composition.png", dpi=130, bbox_inches="tight")

# Figure 2 : individual angles
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for i, (th, lab) in enumerate(angles):
    gcr, bb1 = indiv[lab]
    ax[i].plot(alpha_list, gcr, '-o', color="firebrick", lw=2, ms=4, label="GCR (2 gates)")
    ax[i].plot(alpha_list, bb1, '-s', color="cornflowerblue", lw=2, ms=4, label="BB1 (4 gates)")
    ax[i].set_yscale("log"); ax[i].set_xlabel(r"$|\alpha|$")
    ax[i].set_title(rf"rotation angle $\theta={lab[1:-1]}$")
    ax[i].grid(alpha=0.3); ax[i].legend(fontsize=9)
ax[0].set_ylabel(r"success-heralded infidelity $1-F$")
fig.suptitle("Claim 3: standalone GCR(θ) beats standalone BB1(θ) at every individual angle", fontsize=12)
fig.tight_layout()
fig.savefig("Paper_Figures/check_individual_angles.png", dpi=130, bbox_inches="tight")

# ---- text summary ----
print("\n================ SUMMARY ================")
def meanq(p): return np.mean(np.abs(np.array(p) - np.round(p)))
print("CLAIM 1&2 (composition), mean |P-nearest(0/1)|  (lower=better readout):")
print(f"   bare BB1                 : {meanq(P_bb1):.3f}")
print(f"   BB1(GCR) vec*vecf [nonU] : {meanq(P_non):.3f}   {'<-- better than BB1' if meanq(P_non)<meanq(P_bb1) else ''}")
print(f"   BB1(GCR) vec      [unit] : {meanq(P_uni):.3f}   {'<-- worse than BB1' if meanq(P_uni)>meanq(P_bb1) else ''}")
print("\nCLAIM 3 (individual angles), median success-infidelity over |alpha|:")
for th, lab in angles:
    gcr, bb1 = indiv[lab]
    win = "GCR better" if np.median(gcr) < np.median(bb1) else "BB1 better"
    print(f"   theta={lab:>6}:  GCR={np.median(gcr):.2e}   BB1={np.median(bb1):.2e}   -> {win}")
print("\nsaved Paper_Figures/check_composition.png and Paper_Figures/check_individual_angles.png")
