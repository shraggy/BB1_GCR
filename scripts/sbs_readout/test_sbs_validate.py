"""TASK 0: validate the sBs (small-Big-small) stabilizer convention.

sBs round (per-axis v): CD(lambda,sigma_y) * CD(i*2*alpha,sigma_x) * CD(lambda,sigma_y)
  alpha = sqrt(pi/2)/2, lambda = -alpha*Delta^2, Delta=0.34
dual round (swap x<->p, stabilizes the conjugate quadrature):
  CD(i*lambda,sigma_x) * CD(2*alpha,sigma_y) * CD(i*lambda,sigma_x)

The paper's formal CD def: CD(beta,sigma) = exp(i*k*vhat(x)sigma), vhat = Im(beta)*xhat - Re(beta)*phat.
We sweep the ambiguous prefactor k in {2, sqrt(2), 1} and overall sign, whether rounds alternate
x/p (dual), and ancilla reset state |g> vs |+>, and pick whichever setting drives a displaced
finite-energy GKP |0> monotonically back toward fidelity>0.98 / impurity<1e-2 within <=10 NOISELESS
rounds (with ancilla traced out + reset each round).
"""
import numpy as np
from qutip import *

Ncav = 100
aC = destroy(Ncav); xO = (aC + aC.dag())/np.sqrt(2); pO = (-1j)*(aC - aC.dag())/np.sqrt(2)
Delta = 0.34; r = -np.log(Delta); sq = np.sqrt(np.pi)
alpha = np.sqrt(np.pi/2)/2
lam = -alpha*Delta**2

def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()

L0 = logical(0)
L0dm = ket2dm(L0)
gq = basis(2,0); plus = (basis(2,0)+basis(2,1)).unit()

def CDop(beta, sigma, k, sign):
    vhat = sign*k*(np.imag(beta)*xO - np.real(beta)*pO)
    return (1j*tensor(vhat, sigma)).expm(method='dense')

def make_round_x(k, sign):
    small = CDop(lam, sigmay(), k, sign)
    big = CDop(1j*2*alpha, sigmax(), k, sign)
    return small*big*small

def make_round_p(k, sign):
    small = CDop(1j*lam, sigmax(), k, sign)
    big = CDop(2*alpha, sigmay(), k, sign)
    return small*big*small

def run_trial(k, sign, alternate, anc_state_name, nrounds=10, disp=0.15+0.12j):
    Rx = make_round_x(k, sign)
    Rp = make_round_p(k, sign) if alternate else None
    anc = gq if anc_state_name=='g' else plus
    ancdm = ket2dm(anc)
    psi0 = (displace(Ncav, disp)*L0).unit()
    rho = tensor(ket2dm(psi0), ancdm)
    fids=[]; impurs=[]
    ro0 = rho.ptrace(0)
    fids.append(float(np.real(expect(L0dm, ro0))))
    impurs.append(1-float(np.real((ro0*ro0).tr())))
    for n in range(nrounds):
        R = Rx if (not alternate or n%2==0) else Rp
        rho = R*rho*R.dag()
        ro = rho.ptrace(0)
        # reset ancilla
        rho = tensor(ro, ancdm)
        fids.append(float(np.real(expect(L0dm, ro))))
        impurs.append(1-float(np.real((ro*ro).tr())))
    return fids, impurs

def judge(fids, impurs):
    # success: final fidelity > 0.98, final impurity < 1e-2, and roughly monotonic improvement
    # (not oscillating wildly, not collapsing to zero purity)
    fmax = max(fids); fend=fids[-1]
    if fend < 0.98: return False, f"fend={fend:.3f}"
    if impurs[-1] > 1e-2: return False, f"impur_end={impurs[-1]:.3e}"
    # check monotonic-ish: last 3 values non-decreasing beyond small tol, and no big oscillation (std of diffs vs trend)
    diffs = np.diff(fids)
    n_bad = np.sum(diffs < -0.05)
    if n_bad > 1: return False, f"oscillating n_bad={n_bad}"
    return True, f"fend={fend:.4f} impur_end={impurs[-1]:.3e}"

print(f"alpha={alpha:.6f} lambda={lam:.6f}")
print(f"{'k':>6} {'sign':>5} {'alt':>5} {'anc':>4}  fidelities (0..10 rounds)")
results=[]
for k in [2, np.sqrt(2), 1]:
    for sign in [1,-1]:
        for alternate in [False, True]:
            for anc_state in ['g','+']:
                fids, impurs = run_trial(k, sign, alternate, anc_state)
                ok, msg = judge(fids, impurs)
                results.append((k,sign,alternate,anc_state,fids,impurs,ok,msg))
                tag = "WIN" if ok else "   "
                fidstr = " ".join(f"{f:.3f}" for f in fids)
                print(f"{tag} k={k:.4f} sign={sign:+d} alt={alternate!s:5} anc={anc_state}  F={fidstr}  ({msg})")

wins = [x for x in results if x[6]]
print(f"\n{len(wins)} winning configuration(s) out of {len(results)}")
for k,sign,alternate,anc_state,fids,impurs,ok,msg in wins:
    print(f"WINNER: k={k}, sign={sign}, alternate={alternate}, ancilla={anc_state}")
    print("  fidelities:", [round(f,4) for f in fids])
    print("  impurities:", [f"{im:.2e}" for im in impurs])
