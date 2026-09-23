"""
PART 1 (user's new request): non-unitary BB1(GCR) composition, three variants
   - all4 vec*vecf      (user's fix: last pulse now vec*vecf too)
   - notebook-orig      (stages G4,G3,G2 = vec*vecf ; last stage G1 = vec)
   - all4 vec           (fully unitary, physical)
   + bare BB1 baseline.
   Shows: switching the LAST pulse vec*vecf->vec mildly degrades; switching ALL
   to vec is catastrophic.

PART 2 (claim 3, fair test): standalone GCR(theta) vs BB1(theta) for theta in
   {pi/2, pi, 2pi}, reporting BOTH failure probability P_e and success infidelity.
   (faithful port of cells 697ba552 (GCR) and 26f0d1a7 (BB1)).
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

# ============ PART 1 : composition (i=sqrt(2)) ============
NcavA = 140
aOp = destroy(NcavA); aOp1 = aOp.dag()
xOpA = (aOp + aOp1) / np.sqrt(2)
pOpA = (-1j) * (aOp - aOp1) / np.sqrt(2)
def rot_xyA(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
Delta = 0.34; r = -np.log(Delta)
thetaA = np.pi / 2
phi0 = 0; phi1 = np.arccos(-thetaA / (4 * np.pi)); ep = Delta ** 2
xfac = np.pi / thetaA; beta = thetaA / 2 / (np.sqrt(np.pi) / 2)

def build_stages(last_mode, body_mode):
    """body_mode for stages G4,G3,G2 ; last_mode for stage G1. each 'vecvecf' or 'vec'."""
    def A(v, vf, m): return (v * vf) if m == 'vecvecf' else v
    out = []
    vf = vec_f(sigma_xyz(0, 0), rot_xyA(0, 0)); v = 1j * sigma(phi1) * vf
    out.append(tensor(-1j*xfac*beta*xOpA, sigma(phi1)).expm(method='dense') * tensor(-1j*xfac*beta*(ep*pOpA), A(v, vf, body_mode)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(2*np.pi, phi1)); v = 1j*sigma(3*phi1)*vf
    out.append(tensor(-2j*np.sqrt(np.pi)*xOpA, sigma(3*phi1)).expm(method='dense') * tensor(-2j*np.sqrt(np.pi)*(ep*pOpA), A(v, vf, body_mode)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(4*np.pi, 3*phi1)); v = 1j*sigma(phi1)*vf
    out.append(tensor(-1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense') * tensor(-1j*np.sqrt(np.pi)*(ep*pOpA), A(v, vf, body_mode)).expm(method='dense'))
    vf = vec_f(vf, rot_xyA(2*np.pi, phi1)); v = 1j*sigma(phi0)*vf
    out.append(tensor(-1j*beta*xOpA, sigma(phi0)).expm(method='dense') * tensor(-1j*beta*(ep*pOpA), A(v, vf, last_mode)).expm(method='dense'))
    return out

ST_all = build_stages('vecvecf', 'vecvecf')   # user's new version
ST_nb  = build_stages('vec', 'vecvecf')        # notebook original (last = vec)
ST_uni = build_stages('vec', 'vec')            # fully unitary
B = [tensor(1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense').dag(),
     tensor(2j*np.sqrt(np.pi)*xOpA, sigma(3*phi1)).expm(method='dense').dag(),
     tensor(1j*np.sqrt(np.pi)*xOpA, sigma(phi1)).expm(method='dense').dag(),
     tensor(1j*beta*xOpA, sigma(phi0)).expm(method='dense').dag()]

def run_comp(stages, init):
    s = tensor(init, g)
    for U in stages: s = (U * s).unit()
    return s

print(f"[A setup {time.time()-t0:.0f}s] sweeping composition ...")
NA = 49
alphA = np.linspace(-2*np.sqrt(np.pi), 2*np.sqrt(np.pi), NA)
aoff = -np.sqrt(np.pi) / 2
res = {k: [] for k in ['m', 'P_bb1', 'P_all', 'P_nb', 'P_uni']}
for a1 in alphA:
    alpha = a1 + aoff
    init = (displace(NcavA, alpha/np.sqrt(2)) * (squeeze(NcavA, r) * basis(NcavA, 0)).unit()).unit()
    res['m'].append(a1/np.sqrt(np.pi))
    res['P_bb1'].append(np.real(expect(ket2dm(py), run_comp(B, init).ptrace(1))))
    res['P_all'].append(np.real(expect(ket2dm(py), run_comp(ST_all, init).ptrace(1))))
    res['P_nb'].append(np.real(expect(ket2dm(py), run_comp(ST_nb, init).ptrace(1))))
    res['P_uni'].append(np.real(expect(ket2dm(py), run_comp(ST_uni, init).ptrace(1))))
for k in res: res[k] = np.array(res[k])
np.savez("Paper_Data/checks2_composition.npz", **res)
print(f"[A done {time.time()-t0:.0f}s]")

# ============ PART 2 : individual angles, P_e and infidelity (i=2) ============
NcavB = 140
aOpB = destroy(NcavB); xOpB = (aOpB + aOpB.dag()) / 2; pOpB = (-1j) * (aOpB - aOpB.dag()) / 2
def rot_xyB(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
alpha_list = np.arange(40, 136, 8) / 20.0

def gcr_primitive(theta):
    qub = rx(theta) * g
    fp, sf = [], []
    for alpha in alpha_list:
        chi = -theta / 2 / alpha
        state = (tensor(1j*chi*xOpB, sigmax()).expm(method='dense') * tensor(1j*chi*pOpB, sigmay()).expm(method='dense') * tensor(coherent(NcavB, alpha), g)).unit()
        state2 = tensor(coherent(NcavB, alpha), qub).unit()
        proj = tensor(qeye(NcavB), ket2dm(qub))
        state3 = (proj * state).unit()
        fp.append(1 - expect(proj, state))           # failure probability
        sf.append(1 - np.abs(state3.overlap(state2))**2)  # success infidelity
    return np.array(fp), np.array(sf)

def bb1_primitive(theta):
    phi1 = np.arccos(-theta / (4 * np.pi)); ep = 0
    qub = rx(-theta) * g
    fp, sf = [], []
    for alpha in alpha_list:
        initial = coherent(NcavB, alpha)
        xexp = np.abs(expect(xOpB, initial)); bet = theta / xexp
        state = tensor(g, initial)
        vf = vec_f(sigma_xyz(0, 0), rot_xyB(0, 0)); v = 1j*sigma(0)*vf
        state = (tensor(sigma(0), 1j*bet/2*xOpB).expm(method='dense') * tensor(v, 1j*bet/2*(ep*pOpB)).expm(method='dense') * state).unit()
        vf = vec_f(vf, rot_xyB(np.pi/2, 0)); v = 1j*sigma(phi1)*vf
        state = (tensor(sigma(phi1), 1j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 1j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense') * state).unit()
        vf = vec_f(vf, rot_xyB(2*np.pi, phi1)); v = 1j*sigma(3*phi1)*vf
        state = (tensor(sigma(3*phi1), 2j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 2j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense') * state).unit()
        vf = vec_f(vf, rot_xyB(4*np.pi, 3*phi1)); v = -1j*sigma(phi1)*vf
        state = (tensor(sigma(phi1), 1j/2*np.pi/xexp*xOpB).expm(method='dense') * tensor(v, 1j/2*np.pi/xexp*(ep*pOpB)).expm(method='dense') * state).unit()
        state2 = tensor(qub, coherent(NcavB, alpha))
        proj = tensor(ket2dm(qub), qeye(NcavB))
        state3 = (proj * state).unit()
        fp.append(1 - expect(proj, state))
        sf.append(1 - np.abs(state3.overlap(state2))**2)
    return np.array(fp), np.array(sf)

angles = [(np.pi/2, "pi/2"), (np.pi, "pi"), (2*np.pi, "2pi")]
data = {}
for th, lab in angles:
    print(f"[B {lab} {time.time()-t0:.0f}s] ...")
    data[lab] = (gcr_primitive(th), bb1_primitive(th))
np.savez("Paper_Data/checks2_individual.npz", alpha=alpha_list,
         **{f"gcr_fp_{l}": data[l][0][0] for _, l in angles},
         **{f"gcr_sf_{l}": data[l][0][1] for _, l in angles},
         **{f"bb1_fp_{l}": data[l][1][0] for _, l in angles},
         **{f"bb1_sf_{l}": data[l][1][1] for _, l in angles})
print(f"[B done {time.time()-t0:.0f}s]")

# ============ PLOTS ============
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(res['m'], res['P_all'], '--', color="firebrick", lw=3, label="all-4 vec*vecf (new)")
ax[0].plot(res['m'], res['P_nb'], ':', color="darkorange", lw=2.5, label="notebook: last=vec")
ax[0].plot(res['m'], res['P_bb1'], color="cornflowerblue", lw=2, label="bare BB1")
ax[0].plot(res['m'], res['P_uni'], '-', color="seagreen", lw=2, label="all-4 vec (unitary)")
ax[0].set_xlabel(r"$\langle x\rangle/\sqrt{\pi}$"); ax[0].set_ylabel("P(+1)")
ax[0].set_title("Composition readout response"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
def meanq(p): return np.mean(np.abs(p - np.round(p)))
labels = ['all vec*vecf', 'notebook last=vec', 'bare BB1', 'all vec']
vals = [meanq(res['P_all']), meanq(res['P_nb']), meanq(res['P_bb1']), meanq(res['P_uni'])]
ax[1].bar(range(4), vals, color=["firebrick", "darkorange", "cornflowerblue", "seagreen"])
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, rotation=20, fontsize=8)
ax[1].set_ylabel("mean |P - nearest(0/1)|  (lower=better)")
ax[1].set_title("Readout quality"); ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig("Paper_Figures/check2_composition.png", dpi=130, bbox_inches="tight")

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, (th, lab) in enumerate(angles):
    (gfp, gsf), (bfp, bsf) = data[lab]
    ax[0, i].plot(alpha_list, gfp, '-o', color="firebrick", lw=2, ms=4, label="GCR (2 gates)")
    ax[0, i].plot(alpha_list, bfp, '-s', color="cornflowerblue", lw=2, ms=4, label="BB1 (4 gates)")
    ax[0, i].set_yscale("log"); ax[0, i].set_title(rf"$\theta={lab}$  |  failure prob $P_e$")
    ax[0, i].set_xlabel(r"$|\alpha|$"); ax[0, i].grid(alpha=0.3); ax[0, i].legend(fontsize=8)
    ax[1, i].plot(alpha_list, gsf, '-o', color="firebrick", lw=2, ms=4, label="GCR")
    ax[1, i].plot(alpha_list, bsf, '-s', color="cornflowerblue", lw=2, ms=4, label="BB1")
    ax[1, i].set_yscale("log"); ax[1, i].set_title(rf"$\theta={lab}$  |  success infidelity")
    ax[1, i].set_xlabel(r"$|\alpha|$"); ax[1, i].grid(alpha=0.3); ax[1, i].legend(fontsize=8)
ax[0, 0].set_ylabel(r"$P_e$"); ax[1, 0].set_ylabel(r"$1-F$")
fig.suptitle("Individual angles: GCR vs BB1  (top: failure prob, bottom: success infidelity)", fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/check2_individual_angles.png", dpi=130, bbox_inches="tight")

print("\n================ SUMMARY ================")
print("PART 1 composition, mean |P-nearest(0/1)| (lower=better):")
for l, v in zip(labels, vals): print(f"   {l:>20}: {v:.3f}")
print("\nPART 2 individual angles, median over |alpha|:")
print(f"   {'angle':>6} | {'GCR P_e':>10} {'BB1 P_e':>10} {'win':>9} | {'GCR 1-F':>10} {'BB1 1-F':>10} {'win':>9}")
for th, lab in angles:
    (gfp, gsf), (bfp, bsf) = data[lab]
    wfp = "GCR" if np.median(gfp) < np.median(bfp) else "BB1"
    wsf = "GCR" if np.median(gsf) < np.median(bsf) else "BB1"
    print(f"   {lab:>6} | {np.median(gfp):.2e} {np.median(bfp):.2e} {wfp:>9} | {np.median(gsf):.2e} {np.median(bsf):.2e} {wsf:>9}")
print("\nsaved check2_composition.png and check2_individual_angles.png")
