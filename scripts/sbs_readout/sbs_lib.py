"""Shared sBs (small-Big-small) stabilizer library.

VALIDATED CD CONVENTION (Task 0 result, unchanged):
  CD(beta,sigma) = exp(i*k*(Im(beta)*xhat - Re(beta)*phat) (x) sigma),  k = sqrt(2)
This is the true displacement-operator prefactor; k=sqrt(2) is the unique value (of the swept
{2,sqrt(2),1}x{+/-1}) that makes a single round_x drive a displaced GKP state monotonically back
toward the codeword, with near-identity action (impurity cost ~3e-3/round from ancilla
entanglement only) on the ideal codeword itself.

round_x = CD(lambda,sigma_y) * CD(i*2*alpha,sigma_x) * CD(lambda,sigma_y)   (user-specified, validated)
round_p = CD(-i*lambda,sigma_y) * CD(2*alpha,sigma_x) * CD(-i*lambda,sigma_y)   (corrected dual, per
  coordinator/author: SAME Pauli axes as round_x (sigma_y on smalls, sigma_x on big) -- only the
  real/imaginary role of the CD argument is swapped (small: real->imaginary with a sign flip on
  lambda; big: imaginary->real, no sign flip), NOT a Pauli axis swap. An earlier attempt that also
  swapped sigma_y<->sigma_x for round_p was wrong: it deterministically flipped logical(0)<->
  logical(1) and added ~5-8% impurity/round instead of correcting anything. This version is
  validated in test_round_p_v2.py before being adopted here.

Both rounds decompose (via CD(beta,sigma)=exp(i*k*vhat*sigma), vhat=Im(beta)*xhat-Re(beta)*phat,
gate=exp(-iK) for duration t=|beta|) into 3 gates identified by name: 'x_small','x_big','x_small'
and 'p_small','p_big','p_small'.

PERFORMANCE: for the NOISELESS case, the 4 distinct gate unitaries (x_small/x_big/p_small/p_big)
are PRECOMPUTED AND CACHED as U=exp(-iK) (once per Ncav) -- this is the fix that mattered: the
previous implementation rebuilt the tensor-product generator AND redid a fresh dense expm() every
single call, which was the actual bottleneck (a 10-round noiseless round_x sequence went from
minutes to ~1.2s at Ncav=100).
  NOTE on the noisy path: precomputing a cached SUPEROPERATOR propagator (exp(L*t) from the full
Liouvillian) was tried and abandoned -- the superoperator lives on a space of dimension
(2*Ncav)^2, so its dense matrix exponential costs O(Ncav^6) and is completely infeasible at
Ncav=400 (blew up to 100+s / 10+GB even at Ncav=40 in testing). So for noisy conditions each gate
is applied via a fresh mesolve() call per round (same cost model as run_mesolve_4.py, ~5s/gate at
Ncav=400); only the constant K/t/c_ops objects are cached (avoiding rebuild overhead), not the
propagator itself -- there is no cheaper exact alternative for a Lindblad channel at this
Hilbert-space size.
"""
import numpy as np
from qutip import *

Delta = 0.34
alpha = np.sqrt(np.pi/2)/2
lam = -alpha*Delta**2
Kfac = 2                   # author's convention (was np.sqrt(2); see Test-1 root-cause diagnosis:
                            # k=sqrt(2) gives the big-CD generator coeff sqrt(pi) instead of the
                            # correct stabilizer length 2a=sqrt(2*pi) under k=2 -- off by sqrt(2))
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

_ops_cache = {}
def ops(Ncav):
    if Ncav not in _ops_cache:
        aC = destroy(Ncav)
        xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
        _ops_cache[Ncav] = (xO, pO)
    return _ops_cache[Ncav]

def logical(mu, Ncav):
    xO, pO = ops(Ncav)
    r = -np.log(Delta); a = np.sqrt(np.pi/2)
    psi = 0*basis(Ncav); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()

def _gate_KT(gate_name, Ncav):
    """K (with gate=exp(-iK)) and duration t=|beta| for each of the 4 named sub-gates."""
    xO, pO = ops(Ncav)
    if gate_name == 'x_small':
        K = Kfac*lam*tensor(pO, sigmay());        t = abs(lam)
    elif gate_name == 'x_big':
        K = -(Kfac*2*alpha*tensor(xO, sigmax()));  t = 2*alpha
    elif gate_name == 'p_small':
        K = Kfac*lam*tensor(xO, sigmay());         t = abs(lam)
    elif gate_name == 'p_big':
        K = Kfac*2*alpha*tensor(pO, sigmax());     t = 2*alpha
    else:
        raise ValueError(gate_name)
    return K, t

_AXIS_GATES = {'x': ('x_small','x_big','x_small'), 'p': ('p_small','p_big','p_small')}

_cops_cache = {}
def COPS_for(Ncav, cond):
    key = (Ncav, cond)
    if key in _cops_cache:
        return _cops_cache[key]
    Nc = Ncav
    aCd = destroy(Nc)
    sm = tensor(qeye(Nc), sigmam()); sz = tensor(qeye(Nc), sigmaz()); aa = tensor(aCd, qeye(2))
    table = {'noiseless': None,
             'complete': [np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
             'tr_decay': [np.sqrt(gam)*sm], 'tr_deph': [np.sqrt(gamphi/2)*sz], 'osc_decay': [np.sqrt(kappa)*aa]}
    _cops_cache[key] = table[cond]
    return _cops_cache[key]

_gate_cache = {}
def _get_gate(Ncav, gate_name):
    key = (Ncav, gate_name)
    if key not in _gate_cache:
        _gate_cache[key] = _gate_KT(gate_name, Ncav)
    return _gate_cache[key]

_unitary_cache = {}
def _get_unitary(Ncav, gate_name):
    key = (Ncav, gate_name)
    if key not in _unitary_cache:
        K, t = _get_gate(Ncav, gate_name)
        _unitary_cache[key] = (-1j*K).expm(method='dense')
    return _unitary_cache[key]

def apply_cached_gate(rho, Ncav, cond, gate_name):
    cops = COPS_for(Ncav, cond)
    if cond == 'noiseless' or cops is None:
        U = _get_unitary(Ncav, gate_name)
        return U*rho*U.dag()
    else:
        K, t = _get_gate(Ncav, gate_name)
        return mesolve(K/t, rho, [0, t], c_ops=cops, options=opts).states[-1]

def sbs_round(rho_osc, axis, cond, Ncav, herald=False):
    """Apply one sBs round (axis='x' or 'p') to an oscillator-only density matrix.
    Ancilla is freshly prepared |g>, entangled+corrected via the 3 cached CD-gate propagators
    (small,big,small), then traced out (and reset). If herald=True, instead post-select the
    ancilla on |g> (project+renormalize) and return (rho_osc_new, p_success)."""
    gq = basis(2,0); gqdm = ket2dm(gq)
    rho = tensor(rho_osc, gqdm)
    for gate_name in _AXIS_GATES[axis]:
        rho = apply_cached_gate(rho, Ncav, cond, gate_name)
    if not herald:
        return rho.ptrace(0)
    else:
        Pg = tensor(qeye(Ncav), ket2dm(basis(2,0)))
        rho_g = Pg*rho*Pg
        p = float(np.real(rho_g.tr()))
        ro = (rho_g/p).ptrace(0) if p > 1e-12 else rho.ptrace(0)
        return ro, p

def impurity(rho_osc):
    return 1-float(np.real((rho_osc*rho_osc).tr()))

def run_sbs_sequence(rho_osc, cond, Ncav, nrounds=10, order='x', early_stop=False, tol=1e-4):
    """Run the FULL nrounds sBs sequence (cycling through the axis sequence in `order`, e.g. 'x'
    or 'xp'), saving every intermediate oscillator state and impurity. Since the alternating x/p
    trace can be NON-monotonic (round_p's logical back-action can transiently raise impurity
    before later rounds bring it down further), the "optimal" stopping point is the argmin over
    the full trace, not the first non-decrease -- so we return the state AT that argmin round,
    not necessarily the round-10 state.
    Returns (impur_trace, rho_at_argmin, min_impur, argmin_round, n_p_used) where argmin_round is
    the number of sBs rounds actually applied to reach rho_at_argmin (0..nrounds) and n_p_used is
    how many of those were round_p (needed by callers to track the logical-X Pauli-frame parity
    round_p's back-action induces: an ODD n_p_used means the logical frame is flipped relative to
    before this block, and any subsequent readout on this oscillator must be reinterpreted
    P(+1)<->P(-1) before being combined with earlier reads)."""
    trace = [impurity(rho_osc)]
    states = [rho_osc]
    rho = rho_osc
    for n in range(nrounds):
        axis = order[n % len(order)]
        rho = sbs_round(rho, axis, cond, Ncav)
        trace.append(impurity(rho))
        states.append(rho)
    argmin_round = int(np.argmin(trace))
    minimp = trace[argmin_round]
    rho_opt = states[argmin_round]
    n_p_used = sum(1 for n in range(argmin_round) if order[n % len(order)] == 'p')
    return trace, rho_opt, minimp, argmin_round, n_p_used
