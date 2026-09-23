"""Validate the CORRECTED round_p dual (per coordinator/author):
  round_p = CD(-i*lambda,sigma_y) * CD(2*alpha,sigma_x) * CD(-i*lambda,sigma_y)
(same Pauli axes as round_x -- sigma_y on smalls, sigma_x on big -- only real/imaginary role of
the CD argument is swapped, small gets a sign flip on lambda, big does not).

Tests, all at Ncav=100:
 (i)   round_p on the ideal codeword logical(0): should be near-identity (no logical(0)<->
       logical(1) flip, low impurity/round). Also test the opposite lambda sign (+i*lambda) in
       case the derived sign is wrong, and report which is clean.
 (ii)  round_p should correct a PURE imaginary displacement error (displace(0.2j)*logical(0)):
       fidelity should recover toward ~1 over repeated rounds.
 (iii) round_x-only vs alternating round_x/round_p on a symmetric photon-loss-mixed GKP state
       (impurity ~0.08): report full impurity traces for both, 10 rounds, noiseless.
"""
import numpy as np
from qutip import *
import sbs_lib as sb

Ncav = 100
L0 = sb.logical(0, Ncav)
L1 = sb.logical(1, Ncav)
rho0 = ket2dm(L0)

print("=== (i) round_p on ideal codeword (checking for logical flip / impurity growth) ===")
rho = rho0
for n in range(6):
    F0 = float(np.real(expect(rho0, rho)))
    F1 = float(np.real(expect(ket2dm(L1), rho)))
    print(f"  round {n}: impurity={sb.impurity(rho):.4e}  F(L0)={F0:.4f}  F(L1)={F1:.4f}")
    rho = sb.sbs_round(rho, 'p', 'noiseless', Ncav)

print("\n--- sanity: alternate lambda sign (flip lam -> -lam in p_small/p_big) manually ---")
# build an alternate version of round_p with the opposite sign convention to compare
xO, pO = sb.ops(Ncav)
Ksmall_alt = -sb.Kfac*sb.lam*tensor(pO, sigmay())  # inverted sign, using pO instead re-checked below
# Actually just flip overall sign of both K's (equivalent to lambda -> -lambda AND alpha unaffected
# is not quite it; instead directly test beta = +i*lambda for small, and -2*alpha for big)
def alt_round_p(rho_osc, Ncav):
    xO, pO = sb.ops(Ncav)
    Ksmall = -sb.Kfac*sb.lam*tensor(xO, sigmay()); tsmall = abs(sb.lam)
    Kbig   = -sb.Kfac*2*sb.alpha*tensor(pO, sigmax()); tbig = 2*sb.alpha
    gq = basis(2,0); gqdm = ket2dm(gq)
    rho = tensor(rho_osc, gqdm)
    for K,t in [(Ksmall,tsmall),(Kbig,tbig),(Ksmall,tsmall)]:
        U = (-1j*K).expm(method='dense')
        rho = U*rho*U.dag()
    return rho.ptrace(0)

rho = rho0
for n in range(6):
    F0 = float(np.real(expect(rho0, rho)))
    F1 = float(np.real(expect(ket2dm(L1), rho)))
    print(f"  [alt-sign] round {n}: impurity={sb.impurity(rho):.4e}  F(L0)={F0:.4f}  F(L1)={F1:.4f}")
    rho = alt_round_p(rho, Ncav)

print("\n=== (ii) round_p correcting a pure imaginary displacement error (0.2j) ===")
psi_err = (displace(Ncav, 0.2j)*L0).unit()
rho = ket2dm(psi_err)
for n in range(11):
    F0 = float(np.real(expect(rho0, rho)))
    print(f"  round {n}: impurity={sb.impurity(rho):.4e}  F(L0)={F0:.4f}")
    if n < 10:
        rho = sb.sbs_round(rho, 'p', 'noiseless', Ncav)

print("\n=== (iii) round_x-only vs alternating x/p on photon-loss-mixed GKP (target impurity ~0.08) ===")
aC = destroy(Ncav)
kappa_test = 0.05; dt = 0.05
rho_mixed = rho0; tt = 0.0
while sb.impurity(rho_mixed) < 0.07:
    rho_mixed = mesolve(0*qeye(Ncav), rho_mixed, [0, dt], c_ops=[np.sqrt(kappa_test)*aC], options=sb.opts).states[-1]
    tt += dt
    if tt > 50: break
print(f"decohered impurity = {sb.impurity(rho_mixed):.4e} (t={tt:.2f})")

trace_x, _, min_x = sb.run_sbs_sequence(rho_mixed, 'noiseless', Ncav, nrounds=10, order='x', early_stop=False)
trace_xp, _, min_xp = sb.run_sbs_sequence(rho_mixed, 'noiseless', Ncav, nrounds=10, order='xp', early_stop=False)
print(f"round_x-only trace:  {['%.4e'%v for v in trace_x]}")
print(f"round_x-only min:    {min_x:.4e}")
print(f"alternating xp trace:{['%.4e'%v for v in trace_xp]}")
print(f"alternating xp min:  {min_xp:.4e}")
print("\nDONE")
