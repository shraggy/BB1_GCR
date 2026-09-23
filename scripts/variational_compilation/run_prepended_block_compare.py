"""Numerically test whether prepending a BLOCK of momentum-conditioned corrections in front of bare
BB1 improves the GKP end-of-the-line readout, compared against the two existing GCR-BB1 baselines.

Four schemes (all end with the four BARE BB1 position kicks):
  1. SINGLE       -- BB1U * Cdet(c,nu), the ONE QITE-optimized pre-correction from
                     run_qite_gcrbb1.py.  Params: Paper_Data/qite_gcrbb1_params.npy.  NOT re-optimized.
  2. INTERLEAVED  -- the authoritative BB1(GCR) det. construction from run_qite_params.py:
                     U = K3 C3 K2 C2 K1 C1 K0 C0   (each correction sits immediately BEFORE its kick,
                     i.e. correction-then-kick per kick, exactly the ordering consumed by
                     run_mesolve_4.py's sc_qite() / run_gcrbb1_noise.py's sc_gcrbb1()-style loops).
                     Params: Paper_Data/qite_det_params.npy.  NOT re-optimized.
  3. VARIANT A    -- "momentum-BB1 block, free": U = BB1U * (C3 C2 C1 C0), i.e. FOUR free momentum
                     corrections applied back-to-back in BB1 order, THEN the four bare BB1 kicks.
                     8 free params (a_k, nu_k).  QITE (Powell) optimized from scratch.
  4. VARIANT B    -- "front-loaded equivalent": take the INTERLEAVED corrections C_i and analytically
                     conjugate each through the bare kicks that precede it in time, moving it to the
                     very front:  Pre_i = K_{i-1}...K_0 (Pre_0 = Identity),
                                  C_i^front = Pre_i^dagger . C_i . Pre_i .
                     This C_i^front is in general NOT of the pure exp(-i a pO SIG(nu)) form (the
                     position-kick conjugation mixes x and p and entangles qubit/cavity), so we fit
                     each C_i^front to the nearest single momentum-displacement operator
                     (max |Tr(V^dagger C_i^front)|) to get a seed (a_k,nu_k), then QITE-POLISH all
                     8 params of the SAME front-block-then-bare-BB1 parametrization as Variant A.

All conventions (SIG, BB1 angles/phases, kick unitary, Cdet, logical(mu), b0/b1, objective Pp,
Powell options) are copied EXACTLY from run_qite_gcrbb1.py and run_qite_params.py.  The final
noiseless P(e|eps) readout metric (epsA/epsB windows, shift, flipq convention) is copied EXACTLY
from run_gcrbb1_noise.py, restricted to the noiseless condition (coherent floor only).

Optimization is done at Ncav=80 (fast, matches run_qite_gcrbb1.py/run_qite_params.py); final
readout-error evaluation is done at Ncav=200 (bumped up per spec) reusing the same physical params.
"""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np, time
from qutip import *
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

t0 = time.time()

def SIG(p): return np.cos(p)*sigmax() + np.sin(p)*sigmay()

Delta = 0.34; r = -np.log(Delta); ep = Delta**2; sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
beta = (np.pi/2)/2/(np.sqrt(np.pi)/2)     # = sqrt(pi)/2, cell-54 beta (run_qite_gcrbb1.py convention)
BB1 = [(np.pi, phi1), (2*np.pi, 3*phi1), (np.pi, phi1), (np.pi/2, 0.0)]
theta_list = np.array([th for th, ph in BB1])

def build_ops(Ncav):
    aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
    gq = basis(2, 0); py = (basis(2, 0)+1j*basis(2, 1)).unit()
    def logical(mu):
        psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
        for n in range(-nmax, nmax+1):
            psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav, (2*n+mu)*a)*squeeze(Ncav, r)*basis(Ncav, 0)
        return psi.unit()
    L0 = logical(0)
    KICK = [tensor(-1j*(th/sq)*xO, SIG(ph)).expm(method='dense') for th, ph in BB1]
    BB1U = None
    for K in KICK: BB1U = K if BB1U is None else K*BB1U   # Op4*Op3*Op2*Op1, applied left-to-right in time
    def Cdet(c, nu): return tensor(-1j*c*pO, SIG(nu)).expm(method='dense')
    def Pp(U, psi): return float(np.real(expect(ket2dm(py), (U*tensor(psi, gq)).ptrace(1))))
    return dict(aC=aC, xO=xO, pO=pO, gq=gq, py=py, L0=L0, KICK=KICK, BB1U=BB1U, Cdet=Cdet, Pp=Pp)

# ================= optimization stage @ Ncav=80 =================
NcavOpt = 80
O = build_ops(NcavOpt)
L0, KICK, BB1U, Cdet, Pp = O['L0'], O['KICK'], O['BB1U'], O['Cdet'], O['Pp']
b0 = -0.65; b1 = b0 + np.sqrt(np.pi)/np.sqrt(2)
psi0 = (displace(NcavOpt, b0)*L0).unit(); psi1 = (displace(NcavOpt, b1)*L0).unit()

def obj_from_U(Ufn):
    def obj(p):
        U = Ufn(p); return (1-Pp(U, psi0)) + Pp(U, psi1)
    return obj

POWELL_OPTS = {'maxiter': 200, 'xtol': 1e-5, 'ftol': 1e-10}

# ---- Scheme 1: SINGLE (baseline, load only, no re-optimization) ----
gp_single = np.load('Paper_Data/qite_gcrbb1_params.npy')
def Uof_single(p, BB1U=BB1U, Cdet=Cdet): return BB1U*Cdet(p[0], p[1])
U1 = Uof_single(gp_single)
print(f"[{time.time()-t0:.0f}s] SINGLE (loaded, no re-opt) params={np.round(gp_single,5)} "
      f"err0={1-Pp(U1,psi0):.3e} err1={Pp(U1,psi1):.3e}", flush=True)

# ---- Scheme 2: INTERLEAVED (baseline, load only, no re-optimization) ----
gp_det = np.load('Paper_Data/qite_det_params.npy')
def Uof_det(p, KICK=KICK, Cdet=Cdet):
    U = None
    for i in range(4):
        op = KICK[i]*Cdet(p[2*i], p[2*i+1]); U = op if U is None else op*U
    return U
U2 = Uof_det(gp_det)
print(f"[{time.time()-t0:.0f}s] INTERLEAVED (loaded, no re-opt) params={np.round(gp_det,5)} "
      f"err0={1-Pp(U2,psi0):.3e} err1={Pp(U2,psi1):.3e}", flush=True)

# ---- Scheme 3: VARIANT A -- front BLOCK of 4 free momentum corrections, then bare BB1 ----
def Cblock(p, Cdet=Cdet):
    Cb = None
    for k in range(4):
        C = Cdet(p[2*k], p[2*k+1]); Cb = C if Cb is None else C*Cb
    return Cb
def Uof_block(p, BB1U=BB1U): return BB1U*Cblock(p)

# Seed: axes at the BB1 phases; amplitudes distributed proportional to theta_k, normalized so the
# k=3 (theta=pi/2) piece reproduces the single-correction analytic seed -(beta*ep/4):
#   a_k = -(beta*ep/4) * (theta_k / (pi/2))   for theta_k in [pi, 2pi, pi, pi/2] -> ratios [2,4,2,1]
seedA_a = -(beta*ep/4) * (theta_list/(np.pi/2))
seedA_nu = np.array([phi1, 3*phi1, phi1, 0.0])
seedA = np.empty(8); seedA[0::2] = seedA_a; seedA[1::2] = seedA_nu
print(f"[{time.time()-t0:.0f}s] VARIANT A seed a_k={np.round(seedA_a,5)} nu_k={np.round(seedA_nu,4)}", flush=True)
objA = obj_from_U(Uof_block)
print(f"[{time.time()-t0:.0f}s] VARIANT A seed obj={objA(seedA):.3e}", flush=True)
resA = minimize(objA, seedA, method='Powell',
    callback=lambda xk: print(f'[{time.time()-t0:.0f}s] A iter obj={objA(xk):.3e}', flush=True),
    options=POWELL_OPTS)
paramsA = resA.x; UA = Uof_block(paramsA)
print(f"[{time.time()-t0:.0f}s] VARIANT A OPT params={np.round(paramsA,5)} "
      f"err0={1-Pp(UA,psi0):.3e} err1={Pp(UA,psi1):.3e} obj={resA.fun:.3e}", flush=True)

# ---- Scheme 4: VARIANT B -- front-loaded conjugation of INTERLEAVED corrections, then QITE-polish ----
# Interleaved time order (from run_qite_params.py's Uof): C0,K0,C1,K1,C2,K2,C3,K3
#   => U = K3 C3 K2 C2 K1 C1 K0 C0
# Moving C_i to the very front requires conjugating it through the kicks that acted before it:
#   Pre_i = K_{i-1} ... K_0  (time-ordered product of the kicks preceding kick i; Pre_0 = Identity)
#   C_i^front = Pre_i^dagger . C_i . Pre_i
# (derivation: repeatedly use  X * K = K * (K^dagger X K)  to pull each C_i leftward past the
#  earlier kicks one at a time, which reproduces exactly this sandwich for the i-th correction.)
Iop = tensor(qeye(NcavOpt), qeye(2))
Pre = [Iop]
for i in range(3): Pre.append(KICK[i]*Pre[i])          # Pre[i] = K_{i-1}...K_0
Cdet_i = [Cdet(gp_det[2*i], gp_det[2*i+1]) for i in range(4)]
Cfront_exact = [Pre[i].dag()*Cdet_i[i]*Pre[i] for i in range(4)]

dim = 2*NcavOpt
def fitdist(q, target):
    a, nu = q; V = Cdet(a, nu)
    return -abs((V.dag()*target).tr())/dim            # maximize process overlap -> minimize negative

seedB = np.empty(8)
for i in range(4):
    seed0 = np.array([gp_det[2*i], gp_det[2*i+1]])      # seed the fit at the original det. (c_i,nu_i)
    fr = minimize(fitdist, seed0, args=(Cfront_exact[i],), method='Powell',
                  options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-10})
    seedB[2*i:2*i+2] = fr.x
    print(f"[{time.time()-t0:.0f}s] VARIANT B front-fit k={i}: a={fr.x[0]:+.4f} nu={fr.x[1]:.4f} "
          f"overlap={-fr.fun:.4f} (1.0=exact single-momentum form)", flush=True)

# NOTE on the approximation: C_i^front generally mixes x/p and entangles qubit-cavity (conjugation
# through a position kick does not preserve the pure exp(-i a pO SIG(nu)) form), so the fit overlap
# above is typically well below 1. We use it only as a seed and let Powell polish within the SAME
# front-block-then-bare-BB1 parametrization used for Variant A (documented per spec fallback).
objB = obj_from_U(Uof_block)
print(f"[{time.time()-t0:.0f}s] VARIANT B seed obj={objB(seedB):.3e}", flush=True)
resB = minimize(objB, seedB, method='Powell',
    callback=lambda xk: print(f'[{time.time()-t0:.0f}s] B iter obj={objB(xk):.3e}', flush=True),
    options=POWELL_OPTS)
paramsB = resB.x; UB = Uof_block(paramsB)
print(f"[{time.time()-t0:.0f}s] VARIANT B OPT params={np.round(paramsB,5)} "
      f"err0={1-Pp(UB,psi0):.3e} err1={Pp(UB,psi1):.3e} obj={resB.fun:.3e}", flush=True)

np.savez('Paper_Data/prepended_block_params.npz', variantA_params=paramsA, variantB_params=paramsB)
print(f"[{time.time()-t0:.0f}s] saved Paper_Data/prepended_block_params.npz", flush=True)

# ================= evaluation stage @ Ncav=200 (noiseless coherent floor) =================
NcavEval = 200
E = build_ops(NcavEval)
L0e, KICKe, BB1Ue, Cdete, Ppe = E['L0'], E['KICK'], E['BB1U'], E['Cdet'], E['Pp']

def Uof_single_e(p): return BB1Ue*Cdete(p[0], p[1])
def Uof_det_e(p):
    U = None
    for i in range(4):
        op = KICKe[i]*Cdete(p[2*i], p[2*i+1]); U = op if U is None else op*U
    return U
def Cblock_e(p):
    Cb = None
    for k in range(4):
        C = Cdete(p[2*k], p[2*k+1]); Cb = C if Cb is None else C*Cb
    return Cb
def Uof_block_e(p): return BB1Ue*Cblock_e(p)

U_single = Uof_single_e(gp_single)
U_det = Uof_det_e(gp_det)
U_A = Uof_block_e(paramsA)
U_B = Uof_block_e(paramsB)

u = np.sqrt(np.pi)/(2*np.sqrt(2)); shift = np.sqrt(np.pi/2)/2; NE = 8
epsA = np.linspace(0, u, NE); epsB = np.linspace(2*u, 3*u, NE)

def readout_curve(U, eps_arr, flipq):
    qs = []
    for e in eps_arr:
        psi = (displace(NcavEval, e-shift)*L0e).unit()
        qp = 1 - Ppe(U, psi)
        qs.append((1-qp) if flipq else qp)
    return np.array(qs)

ORDER = ['SINGLE', 'INTERLEAVED', 'VARIANT_A', 'VARIANT_B']
SCHEMES = {'SINGLE': U_single, 'INTERLEAVED': U_det, 'VARIANT_A': U_A, 'VARIANT_B': U_B}
PARAMS = {'SINGLE': gp_single, 'INTERLEAVED': gp_det, 'VARIANT_A': paramsA, 'VARIANT_B': paramsB}
RESULTS = {}
print(f"[{time.time()-t0:.0f}s] === EVALUATION @ Ncav={NcavEval}, NE={NE}, noiseless ===", flush=True)
for name in ORDER:
    U = SCHEMES[name]
    qa = readout_curve(U, epsA, False)
    qb = readout_curve(U, epsB, True)
    RESULTS[name] = dict(qa=qa, qb=qb)
    print(f"[{time.time()-t0:.0f}s] {name:12s} params={np.round(PARAMS[name],5)}", flush=True)
    print(f"    logical-0 (eps in [0,u]):  err(eps=0)={qa[0]:.3e}   mean={qa.mean():.3e}", flush=True)
    print(f"    logical-1 (eps in [2u,3u]): err(eps=2u)={qb[0]:.3e}  mean={qb.mean():.3e}", flush=True)

# ================= plot =================
_rob = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family': 'Roboto', 'mathtext.fontset': 'cm', 'axes.labelsize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 1.6,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.fontsize': 10})
def clip(v): return np.clip(np.abs(v), 1e-8, None)
COL = {'SINGLE': 'teal', 'INTERLEAVED': 'darkviolet', 'VARIANT_A': 'firebrick', 'VARIANT_B': 'darkorange'}
LAB = {'SINGLE': 'GCR-BB1 (1 pre-corr.)', 'INTERLEAVED': 'BB1(GCR) det. (interleaved)',
       'VARIANT_A': 'prepended block, free (A)', 'VARIANT_B': 'prepended block, front-loaded (B)'}
COLS = [('logical-0', epsA, 0, u, [0, u], [r"$0$", r"$\frac{\sqrt{\pi}}{2\sqrt{2}}$"], 'qa'),
        ('logical-1', epsB, 2*u, 3*u, [2*u, 3*u],
         [r"$\frac{\sqrt{\pi}}{\sqrt{2}}$", r"$\frac{3\sqrt{\pi}}{2\sqrt{2}}$"], 'qb')]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for j, (title, eps, lo, hi, ticks, tlab, key) in enumerate(COLS):
    a = ax[j]
    for name in ORDER:
        a.plot(eps, clip(RESULTS[name][key]), '-', color=COL[name], lw=1.8, marker='o', ms=3,
               label=LAB[name] if j == 0 else None)
    a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(1e-8, 1); a.grid(alpha=0.3, which='both')
    a.set_xticks(ticks); a.set_xticklabels(tlab)
    a.set_xlabel(r'displacement error $\epsilon$'); a.set_title(title, fontsize=12)
    if j == 0: a.set_ylabel(r'readout error $P(e|\epsilon)$')
fig.suptitle('Prepended momentum-block vs single/interleaved GCR-BB1 (noiseless coherent floor)', fontsize=13)
fig.legend(loc='lower center', ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.08, 1, 0.96])
fig.savefig('Paper_Figures/prepended_block_compare.pdf', bbox_inches='tight')
fig.savefig('Paper_Figures/prepended_block_compare.png', dpi=150, bbox_inches='tight')
print(f"[{time.time()-t0:.0f}s] wrote Paper_Figures/prepended_block_compare.pdf/.png", flush=True)

# ================= summary =================
def noerr(name): return RESULTS[name]['qa'][0], RESULTS[name]['qb'][0]
def meanerr(name): return RESULTS[name]['qa'].mean(), RESULTS[name]['qb'].mean()
def worst(name): a, b = noerr(name); return max(a, b)

lines = []
lines.append("Prepended momentum-block comparison -- noiseless coherent floor, Ncav=200, NE=8")
lines.append(f"(no-error points: eps=0 in logical-0 window, eps=2u in logical-1 window; u=sqrt(pi)/(2 sqrt2)={u:.6f})")
lines.append("")
for name in ORDER:
    e0, e1 = noerr(name); m0, m1 = meanerr(name)
    lines.append(f"{name:12s} params={np.round(PARAMS[name],5).tolist()}")
    lines.append(f"    no-error err:  logical-0={e0:.3e}   logical-1={e1:.3e}")
    lines.append(f"    mean err:      logical-0={m0:.3e}   logical-1={m1:.3e}")
lines.append("")

w_single, w_inter, w_A, w_B = worst('SINGLE'), worst('INTERLEAVED'), worst('VARIANT_A'), worst('VARIANT_B')
best_variant = 'VARIANT_A' if w_A <= w_B else 'VARIANT_B'
w_best = min(w_A, w_B)
beats_single = w_best < w_single
close_to_interleaved = w_best <= 3*w_inter   # "match" := within a factor of 3 (same order of magnitude)

lines.append("Verdict metric: worst-case no-error readout error max(err@eps=0, err@eps=2u) per scheme:")
lines.append(f"    SINGLE (baseline)       = {w_single:.3e}")
lines.append(f"    INTERLEAVED (baseline)  = {w_inter:.3e}")
lines.append(f"    VARIANT A (free block)  = {w_A:.3e}")
lines.append(f"    VARIANT B (front-load.) = {w_B:.3e}")
lines.append("")
verdict = (f"VERDICT: best prepended block is {best_variant} (worst-case no-error err={w_best:.3e}). "
           f"It {'BEATS' if beats_single else 'does NOT beat'} the single-displacement GCR-BB1 "
           f"({w_single:.3e}) at the no-error point, and it "
           f"{'is within 3x of (roughly MATCHES)' if close_to_interleaved else 'does NOT match'} "
           f"the interleaved BB1(GCR) det. ({w_inter:.3e}).")
lines.append(verdict)
print(verdict, flush=True)

with open('/tmp/prepended_block_summary.txt', 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"[{time.time()-t0:.0f}s] wrote /tmp/prepended_block_summary.txt", flush=True)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
