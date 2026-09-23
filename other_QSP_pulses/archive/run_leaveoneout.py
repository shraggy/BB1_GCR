"""
Is the BB1(GCR) failure specifically due to the 2*pi pulse?

The four stages (application order G4,G3,G2,G1) correspond to rotation angles:
    G4 -> pi,  G3 -> 2*pi,  G2 -> pi,  G1 -> pi/2 (target).
So G3 is THE 2*pi pulse.

We precompute every stage in BOTH forms (non-unitary vec*vecf and unitary vec),
then sweep the composition for configurations that flip ONE stage at a time:
  - from the good chain (all vec*vecf), turn each single stage unitary  -> which hurts most?
  - from the bad chain  (all vec),      turn each single stage non-unit  -> which helps most?
Metric: mean |P(+1) - nearest(0/1)| over the readout sweep (lower = better).
"""
import time
import numpy as np
from qutip import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
g = basis(2, 0); e = basis(2, 1)
py = (g + 1j * e).unit()
def sigma(phi): return np.cos(phi) * sigmax() + np.sin(phi) * sigmay()
def sigma_xyz(th, phi): return np.cos(th) * sigmaz() + np.sin(th) * sigma(phi)
def vec_f(v, R): return R * v * R.dag()

Ncav = 140
aOp = destroy(Ncav); aOp1 = aOp.dag()
xOp = (aOp + aOp1) / np.sqrt(2)
pOp = (-1j) * (aOp - aOp1) / np.sqrt(2)
def rot_xy(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
Delta = 0.34; r = -np.log(Delta)
theta = np.pi / 2
phi0 = 0; phi1 = np.arccos(-theta / (4 * np.pi)); ep = Delta ** 2
xfac = np.pi / theta; beta = theta / 2 / (np.sqrt(np.pi) / 2)

# precompute each stage's x-kick and BOTH precorrection forms
def stage(xkick_coeff, sphi, v, vf):
    xk = tensor(xkick_coeff * xOp, sphi).expm(method='dense')
    pc_nonU = tensor(xkick_coeff * (ep * pOp), v * vf).expm(method='dense')  # vec*vecf
    pc_U = tensor(xkick_coeff * (ep * pOp), v).expm(method='dense')          # vec
    return xk * pc_nonU, xk * pc_U

vf = vec_f(sigma_xyz(0, 0), rot_xy(0, 0)); v = 1j * sigma(phi1) * vf
G4n, G4u = stage(-1j * xfac * beta, sigma(phi1), v, vf)             # pi
vf = vec_f(vf, rot_xy(2*np.pi, phi1)); v = 1j*sigma(3*phi1)*vf
G3n, G3u = stage(-2j * np.sqrt(np.pi), sigma(3*phi1), v, vf)        # 2*pi  <-- the 2pi pulse
vf = vec_f(vf, rot_xy(4*np.pi, 3*phi1)); v = 1j*sigma(phi1)*vf
G2n, G2u = stage(-1j * np.sqrt(np.pi), sigma(phi1), v, vf)          # pi
vf = vec_f(vf, rot_xy(2*np.pi, phi1)); v = 1j*sigma(phi0)*vf
G1n, G1u = stage(-1j * beta, sigma(phi0), v, vf)                    # pi/2
NON = [G4n, G3n, G2n, G1n]
UNI = [G4u, G3u, G2u, G1u]
B = [tensor(1j*np.sqrt(np.pi)*xOp, sigma(phi1)).expm(method='dense').dag(),
     tensor(2j*np.sqrt(np.pi)*xOp, sigma(3*phi1)).expm(method='dense').dag(),
     tensor(1j*np.sqrt(np.pi)*xOp, sigma(phi1)).expm(method='dense').dag(),
     tensor(1j*beta*xOp, sigma(phi0)).expm(method='dense').dag()]
names = ['G4(pi)', 'G3(2pi)', 'G2(pi)', 'G1(pi/2)']
print(f"[setup {time.time()-t0:.0f}s]")

def run(stages, init):
    s = tensor(init, g)
    for U in stages: s = (U * s).unit()
    return s

N = 31
alphA = np.linspace(-2*np.sqrt(np.pi), 2*np.sqrt(np.pi), N)
aoff = -np.sqrt(np.pi) / 2
inits = [(displace(Ncav, (a1+aoff)/np.sqrt(2)) * (squeeze(Ncav, r) * basis(Ncav, 0)).unit()).unit() for a1 in alphA]

def quality(stages):
    P = np.array([np.real(expect(ket2dm(py), run(stages, it).ptrace(1))) for it in inits])
    return np.mean(np.abs(P - np.round(P)))

configs = {}
configs['bare BB1'] = quality(B)
configs['all vec*vecf (good)'] = quality(NON)
configs['all vec (bad)'] = quality(UNI)
# from GOOD: turn one stage unitary
for k in range(4):
    st = list(NON); st[k] = UNI[k]
    configs[f'good, but {names[k]}=vec'] = quality(st)
# from BAD: turn one stage non-unitary
for k in range(4):
    st = list(UNI); st[k] = NON[k]
    configs[f'bad, but {names[k]}=vec*vecf'] = quality(st)
    print(f"[{time.time()-t0:.0f}s] done config set {k}")

print("\n================ LEAVE-ONE-OUT (metric: mean |P-nearest(0/1)|, lower=better) ================")
for k, v in configs.items():
    print(f"   {k:<30}: {v:.3f}")

# attribution
print("\nFrom the GOOD chain (0.10ish), damage from making ONE stage unitary:")
base_good = configs['all vec*vecf (good)']
dmg = {names[k]: configs[f'good, but {names[k]}=vec'] - base_good for k in range(4)}
for n, d in sorted(dmg.items(), key=lambda kv: -kv[1]):
    print(f"   make {n:<10} unitary -> +{d:.3f} worse")
print("\nFrom the BAD chain (0.40ish), repair from making ONE stage non-unitary:")
base_bad = configs['all vec (bad)']
rep = {names[k]: base_bad - configs[f'bad, but {names[k]}=vec*vecf'] for k in range(4)}
for n, d in sorted(rep.items(), key=lambda kv: -kv[1]):
    print(f"   make {n:<10} non-unitary -> -{d:.3f} better")

# plot
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
gd = [configs[f'good, but {names[k]}=vec'] for k in range(4)]
bd = [configs[f'bad, but {names[k]}=vec*vecf'] for k in range(4)]
xk = np.arange(4)
ax[0].bar(xk, gd, color=['gray','firebrick','gray','gray'])
ax[0].axhline(base_good, ls='--', color='green', label='all vec*vecf (good)')
ax[0].set_xticks(xk); ax[0].set_xticklabels(names, rotation=15); ax[0].legend()
ax[0].set_title("Make ONE stage unitary (from good chain)\nhigher bar = that stage matters more")
ax[0].set_ylabel("mean |P - nearest(0/1)|")
ax[1].bar(xk, bd, color=['gray','firebrick','gray','gray'])
ax[1].axhline(base_bad, ls='--', color='red', label='all vec (bad)')
ax[1].set_xticks(xk); ax[1].set_xticklabels(names, rotation=15); ax[1].legend()
ax[1].set_title("Make ONE stage non-unitary (from bad chain)\nlower bar = that stage matters more")
ax[1].set_ylabel("mean |P - nearest(0/1)|")
fig.suptitle("Which stage drives the BB1(GCR) failure? (G3 = the 2π pulse, red)", fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/leave_one_out.png", dpi=130, bbox_inches="tight")
print("\nsaved Paper_Figures/leave_one_out.png")
