"""Fresh stabilized codeword -> m rounds of NOISY authentic SBS -> readout (7 protocols).
LOGICAL-0 ONLY. Ncav=200.

PRIMITIVE: the AUTHENTIC grid()/Hamx/Hamp mesolve SBS (i=2 convention), verbatim from
run_sbs_readout.py (computeTrajCooling, ~line 66-81) and run_sbs_full_sweep.py (standalone
sbs_round(), ~line 77-81). We deliberately do NOT use sbs_lib.sbs_round (that primitive
collapses the ideal codeword after a single noiseless round -- see run_sbs_authentic_phase0.py
for the validation of THIS primitive instead, which is stable).

PHASE 0 FINDING (see run_sbs_authentic_phase0.py output): each single call to sbs_round()
(one grid() + one Urot-conjugated grid(), i.e. one full x-then-p stabilization) FLIPS the
logical parity: F(sbs_round(G_sBs0), G_sBs0)~0.43, F(sbs_round(G_sBs0), G_sBs1)~1.0, and
this is a STABLE, RECOVERING 2-cycle (not a collapse) -- calling sbs_round() again returns to
~0.98+ fidelity with G_sBs0. This matches run_sbs_readout.py's own header comment ("G_sBs1 =
stateList[-1] mu=1-like; G_sBs0 = stateList[-2] mu=0-like") and run_sbs_full_sweep.py's own
convention of applying sbs_round() a literal (even) NROUNDS count. Since this task's M_LIST is
all-even, we apply sbs_round_noisy() a literal m times for each m -- this always lands on the
correct (G_sBs0-parity) branch. Only new physics-facing change vs the original grid(): a `cops`
argument threaded into its 3 existing mesolve() calls (documented QuTiP collapse operators).

State prep (m=0 input): G_sBs0 = computeTrajCooling(Mmax=20) NOISELESS cooling, verbatim recipe
from run_sbs_readout.py lines 66-81.

Readout bridge: run_sbs_readout.py's own i=sqrt(2)-convention scheme builders / run_scheme_dm /
calibrate_D (verbatim), PLUS sc_gcrbb1 and sc_block(paramsA/paramsB) ported verbatim from
recompute_damage_infid.py (lines 57-66) to reach all 7 canonical PROTOCOLS (line 76 there).
D is calibrated ONCE per protocol at beta=0.0 under 'noiseless' on the m=0 state G_sBs0 --
exactly run_sbs_readout.py's calibrate_D() recipe (always noiseless/beta=0 regardless of the
readout condition under test) -- NOT recompute_damage_infid.py's per-condition calibration.
No displacement-error eps is injected (beta=0 throughout): we only care about m-round damage
and readout-time noise, not an eps-robustness sweep.

Metrics per (SBS-cond, m, proto, readout-cond), L0 only:
  P(e)  = qerr = 1 - Re<pydm|rho_readout.ptrace(1)>            (misread probability)
  1-F   = 1 - fidelity(G_sBs0, D_corr * rho_readout.ptrace(0) * D_corr.dag())
          (canonical convention: pristine m=0 codeword reference, D-corrected readout output --
           matches recompute_damage_infid.py's infid=1-fidelity(PREF[stname], D*ro_out*D.dag()))

SBS-noise conditions (final sweep, 5): complete, tr_decay, tr_deph, osc_decay, osc_deph.
  ('noiseless' SBS is a CONTROL only, available here for --preflight-only, never in the full sweep.)
Readout-noise conditions (6): noiseless, complete, tr_decay, tr_deph, osc_decay, osc_deph.
M_LIST = [0,2,4,6,8,10].  Checkpointed to Paper_Data/sbs_authentic_sweep.npz after each
(SBS-cond, proto) pair (d=dict(np.load(path)); d.update(new); np.savez(path,**d) idiom).
"""
import os, sys, time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

Ncav = 200
Mmax = 20
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.; kappa_phi = 1/5000.

# ================= Part 1: authentic sBs, i=2 convention (verbatim, run_sbs_readout.py lines 30-81) =================
g = basis(2,0); e = basis(2,1); px = (g+e).unit()
aOp = destroy(Ncav); aOp1 = aOp.dag()
xOp_s = (aOp+aOp1)/2; pOp_s = (-1j)*(aOp-aOp1)/2

def logical1(mu, delta, N, Ncav, normalize=True):
    psi = 0*basis(Ncav); r = -np.log(delta); a = np.sqrt(np.pi/2)
    for n in range(-int((N+mu)/2), int((N+mu)/2)-mu+1):
        psi = psi + np.sqrt(sp.special.comb(N, n+mu+int(N/2)))*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    if normalize: psi = psi/psi.norm()
    return psi

Delta = 0.34; N = int(np.floor(0.32/Delta**2)); measIndex = Mmax-4
a_sbs = 2*np.sqrt(np.pi); a1 = a_sbs/2
Urot = (1j*(np.pi/2)*aOp1*aOp).expm(); Urot1 = Urot.dag()
Hamx = tensor(sigmaz(), a1*xOp_s); Hamp = tensor(sigmaz(), -a1*pOp_s)
Rx = tensor(rx(np.pi/2), qeye(Ncav)); Rx1 = Rx.dag()
DeltaEpsilon = Delta

def grid(state, qubit, cops=[]):
    """AUTHENTIC sBs stabilization round-component. ONLY new physics-facing change vs.
    run_sbs_readout.py/run_sbs_full_sweep.py: `cops` threaded into the 3 existing mesolve()
    calls (was hardcoded c_ops=[] there)."""
    measState = tensor(qubit, state)
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=cops, options=Options(nsteps=5000)).states[-1]
    measState = Rx*measState*Rx1
    measState = mesolve(Hamp, measState, [0, np.sqrt(2)*np.cosh(Delta**2)], c_ops=cops, options=Options(nsteps=5000)).states[-1]
    measState = Rx1*measState*Rx
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=cops, options=Options(nsteps=10000)).states[-1]
    return measState.ptrace(1).unit(), measState.ptrace(0).unit()

def applyMeasurement(state, correctionDirection):
    displacePauli = displace(Ncav, a_sbs/4/np.sqrt(2)*np.sinh(DeltaEpsilon**2)+1j*a_sbs/4/np.sqrt(2)*np.cosh(DeltaEpsilon**2))
    displacePaulid = displacePauli.dag()
    M = 0.5*(displacePauli+displacePaulid)
    M = displacePauli*M if correctionDirection == 1 else displacePaulid*M
    newState = M*state*M.dag()
    return newState/newState.tr()

def computeTrajCooling(tempState, Mmax):
    """NOISELESS cooling only (used once, to build the m=0 fresh codeword)."""
    stateList = [tempState]
    for i in range(Mmax):
        tempState, tempqubit = grid(tempState, ket2dm(px), [])
        tempState1, tempqubit = grid(Urot*tempState*Urot1, ket2dm(px), [])
        tempState = Urot1*tempState1*Urot
        if i == measIndex:
            tempState = applyMeasurement(tempState, correctionDirection=1)
            tempState = applyMeasurement(tempState, correctionDirection=-1)
        stateList.append(tempState)
    return stateList

# ---- SBS-round noise c_ops table. grid()'s tensor order is (QUBIT, OSC) -- index0=qubit,
# index1=osc (see run_authentic_sbs.py header) -- OPPOSITE of the readout's (OSC,QUBIT) order
# below, so these operators are built independently in the correct order for this subspace.
# Rates match the rest of the repo (gam=gamphi=1/200, kappa=1/1000) plus kappa_phi=1/5000 for
# osc_deph, matching compute_damage_noisy_conds.py's convention (nn=a.dag()*a on the joint
# osc-ancilla space this round acts on; here that's aa_sbs.dag()*aa_sbs on (QUBIT,OSC)). ----
sm_sbs = tensor(sigmam(), qeye(Ncav)); sz_sbs = tensor(sigmaz(), qeye(Ncav)); aa_sbs = tensor(qeye(2), aOp)
nn_sbs = aa_sbs.dag()*aa_sbs
COPS_SBS = {
    'noiseless': [],
    'complete':  [np.sqrt(gam)*sm_sbs, np.sqrt(gamphi/2)*sz_sbs, np.sqrt(kappa)*aa_sbs],
    'tr_decay':  [np.sqrt(gam)*sm_sbs],
    'tr_deph':   [np.sqrt(gamphi/2)*sz_sbs],
    'osc_decay': [np.sqrt(kappa)*aa_sbs],
    'osc_deph':  [np.sqrt(kappa_phi)*nn_sbs],
}
SBS_CONDS = ['complete', 'tr_decay', 'tr_deph', 'osc_decay', 'osc_deph']   # final sweep (5, no noiseless)

def sbs_round_noisy(tempState, sbscond):
    """One authentic SBS round (grid + Urot-conjugated grid) under noise condition `sbscond`.
    Per PHASE 0: this flips logical parity each call; only even call-counts (our whole M_LIST)
    land back on the G_sBs0-parity branch."""
    cops = COPS_SBS[sbscond]
    tempState, _ = grid(tempState, ket2dm(px), cops)
    tempState1, _ = grid(Urot*tempState*Urot1, ket2dm(px), cops)
    tempState = Urot1*tempState1*Urot
    return tempState

M_LIST = [0, 2, 4, 6, 8, 10]

def build_chain(state0, sbscond, m_list=M_LIST):
    """m -> oscillator density matrix after literally m calls to sbs_round_noisy (all m in
    m_list are even, so each returned state is on the G_sBs0-parity branch). Built
    incrementally so the state at m is reused for m+2."""
    chain = {0: state0}
    rho = state0
    m = 0
    for target in sorted(m_list):
        if target == 0:
            continue
        while m < target:
            rho = sbs_round_noisy(rho, sbscond)
            m += 1
        chain[m] = rho
    return chain

log(f"Ncav={Ncav} building G_sBs0 (m=0 input) via NOISELESS computeTrajCooling(Mmax={Mmax})...")
stateList = computeTrajCooling(ket2dm(basis(Ncav, 0)), Mmax)
G_sBs1 = stateList[-1]
G_sBs0 = stateList[-2]
state_0 = ket2dm(logical1(0, Delta, N, Ncav)); state_1 = ket2dm(logical1(1, Delta, N, Ncav))
log(f"G_sBs0 fid to binomial-L0={fidelity(G_sBs0,state_0):.4f}  G_sBs1 fid to binomial-L1={fidelity(G_sBs1,state_1):.4f}")

# ================= Part 2: readout scheme, i=sqrt(2) convention (verbatim, run_sbs_readout.py lines 86-143) =================
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); pyq = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(pyq)
ep = Delta**2; sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
u = np.sqrt(np.pi)/(2*np.sqrt(2)); shift = np.sqrt(np.pi/2)/2
qp = np.load('Paper_Data/qite_det_params.npy')
gp = np.load('Paper_Data/qite_gcrbb1_params.npy')
_blk = np.load('Paper_Data/prepended_block_params.npz')
paramsA = _blk['variantA_params']; paramsB = _blk['variantB_params']
BB1 = [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph = BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph = BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def Uu(K): return (-1j*K).expm(method='dense')
def sc_gcr():
    K,t = det_corr((np.pi/2)*ep/sq, np.pi/2); Kk,tk = kick(3)
    return [('U',Uu(K),K,t),('U',Uu(Kk),Kk,tk)]
def sc_bb1():
    L = []
    for i in range(4): Kk,tk = kick(i); L.append(('U',Uu(Kk),Kk,tk))
    return L
def sc_qite():
    L = []
    for i in range(4):
        K,t = det_corr(qp[2*i],qp[2*i+1]); Kk,tk = kick(i)
        L += [('U',Uu(K),K,t),('U',Uu(Kk),Kk,tk)]
    return L
def sc_herald():
    L = []
    for i in range(4):
        Kk,tk = kick(i)
        if i < 3: L.append(('M',Mexact(i),None,0.0))
        else: K,t = det_corr((np.pi/2)*ep/sq,np.pi/2); L.append(('U',Uu(K),K,t))
        L.append(('U',Uu(Kk),Kk,tk))
    return L
# ---- ported verbatim from recompute_damage_infid.py lines 57-66 (the 3 builders missing from
# run_sbs_readout.py), using THIS script's own kick/det_corr/Uu (identical definitions, same
# i=sqrt(2) convention) ----
def sc_gcrbb1():
    L = []
    Kc,tc = det_corr(gp[0],gp[1]); L.append(('U',Uu(Kc),Kc,tc))
    for i in range(4): Kk,tk = kick(i); L.append(('U',Uu(Kk),Kk,tk))
    return L
def sc_block(p):
    L = []
    for k in range(4): Kc,tc = det_corr(p[2*k],p[2*k+1]); L.append(('U',Uu(Kc),Kc,tc))
    for i in range(4): Kk,tk = kick(i); L.append(('U',Uu(Kk),Kk,tk))
    return L
PROTOCOLS = ['GCR','BB1(GCR)-QITE','BB1','GCR-BB1','GCR-BB1-blockA','GCR-BB1-blockB','herald-RUS']
def build_scheme(name):
    return {'GCR':sc_gcr, 'BB1(GCR)-QITE':sc_qite, 'BB1':sc_bb1, 'GCR-BB1':sc_gcrbb1,
            'GCR-BB1-blockA':lambda:sc_block(paramsA), 'GCR-BB1-blockB':lambda:sc_block(paramsB),
            'herald-RUS':sc_herald}[name]()

sm = tensor(qeye(Ncav),sigmam()); sz = tensor(qeye(Ncav),sigmaz()); aa = tensor(aC,qeye(2))
nn = aa.dag()*aa
COPS_READOUT = {
    'noiseless': None,
    'complete':  [np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
    'tr_decay':  [np.sqrt(gam)*sm],
    'tr_deph':   [np.sqrt(gamphi/2)*sz],
    'osc_decay': [np.sqrt(kappa)*aa],
    'osc_deph':  [np.sqrt(kappa_phi)*nn],
}
READ_CONDS = ['noiseless','complete','tr_decay','tr_deph','osc_decay','osc_deph']
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

def run_scheme_dm(scheme, rho_in, beta, cond):
    """Verbatim from run_sbs_readout.py lines 124-136."""
    D = displace(Ncav, beta)
    rho_osc_err = D*rho_in*D.dag()
    rho = tensor(rho_osc_err, ket2dm(gq)); cops = COPS_READOUT[cond]
    for kind,op,K,t in scheme:
        if kind == 'M':
            rho = op*rho*op.dag(); rho = rho/rho.tr()
        elif cops is None:
            rho = op*rho*op.dag()
        else:
            rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    qerr = 1 - float(np.real(expect(pydm, rho.ptrace(1))))
    return rho_osc_err, rho.ptrace(0), qerr

def calibrate_D(scheme, rho0_in):
    """Verbatim from run_sbs_readout.py lines 138-143: ALWAYS beta=0, ALWAYS 'noiseless',
    regardless of the readout condition under test."""
    rho_osc_err0, ro0, _ = run_scheme_dm(scheme, rho0_in, 0.0, 'noiseless')
    back_x = float(np.real(expect(xO, ro0)))/np.sqrt(2)
    back_p = float(np.real(expect(pO, ro0)))/np.sqrt(2)
    corr = 1j*(back_p + np.sqrt(np.pi/2)/2) - back_x
    return displace(Ncav, corr), back_x, back_p, corr

# ---- FIX (found during Phase 2 preflight, see report): run_sbs_readout.py's own epsA sweep
# uses beta=eps-shift, and since u==shift exactly, eps=0 -> beta=-shift is the "operating
# point" with SMALL qerr (0.0813 for BB1(GCR), their own saved Paper_Data/sbs_readout_fidelity
# .npz), while eps=u -> beta=0 is the AMBIGUOUS endpoint with qerr~=0.5 (verified: feeding
# G_sBs0 with beta=0.0 into run_scheme_dm reproduces their qerr@u=0.50001143 almost exactly).
# So the readout circuit's i=sqrt(2) convention expects a state PRE-SHIFTED by -shift relative
# to the raw noiseless-cooled comb G_sBs0 -- exactly mirroring recompute_damage_infid.py's
# L0=displace(-shift)*logical(0) (shifted, fed to readout) vs L0raw=logical(0) (unshifted,
# used ONLY for calibrate_D). calibrate_D above is correctly left at beta=0.0 (unshifted,
# matches L0raw's role); the ACTUAL per-m readout calls below must use beta=-shift, matching
# run_sbs_readout.py's own eps=0 point (NOT beta=0.0, which is invented/wrong and gives the
# ~0.5 ambiguous floor). The pristine infidelity REFERENCE is correspondingly the SHIFTED
# G_sBs0, G_sBs0_shifted = D(-shift)*G_sBs0*D(-shift).dag(), analogous to PREF['L0'].
G_sBs0_shifted = (displace(Ncav, -shift)*G_sBs0*displace(Ncav, -shift).dag())

# =====================================================================================
def preflight():
    log("=== PRE-FLIGHT: SBS-cond in {noiseless(control),complete}, readout-cond=noiseless, "
        "protocols={GCR,BB1(GCR)-QITE,BB1}, m in [0,2,4] ===")
    pf_sbs_conds = ['noiseless', 'complete']
    pf_protos = ['GCR', 'BB1(GCR)-QITE', 'BB1']
    pf_m = [0, 2, 4]
    results = {}
    for sbscond in pf_sbs_conds:
        log(f"building L0 SBS chain, sbscond={sbscond}, m={pf_m}...")
        chain = build_chain(G_sBs0, sbscond, pf_m)
        for proto in pf_protos:
            scheme = build_scheme(proto)
            D_corr, bx, bp, corr = calibrate_D(scheme, G_sBs0)
            errs = []; infids = []
            for m in pf_m:
                rho_osc_err, ro, qerr = run_scheme_dm(scheme, chain[m], -shift, 'noiseless')
                ro_corr = D_corr*ro*D_corr.dag()
                infid = 1 - float(fidelity(G_sBs0_shifted, ro_corr))
                errs.append(qerr); infids.append(infid)
            results[f'{sbscond}|{proto}|err'] = np.array(errs)
            results[f'{sbscond}|{proto}|infid'] = np.array(infids)
            log(f"  sbscond={sbscond:10s} {proto:16s} P(e) over m={pf_m}: "
                f"{np.array2string(np.array(errs), precision=6)}   "
                f"1-F over m={pf_m}: {np.array2string(np.array(infids), precision=6)}")
    return results, pf_m, pf_sbs_conds, pf_protos

if __name__ == '__main__':
    if '--preflight-only' in sys.argv or '--full' not in sys.argv:
        pf, pf_m, pf_sbs_conds, pf_protos = preflight()
        log("PRE-FLIGHT DONE")
        if '--preflight-only' in sys.argv:
            sys.exit(0)

    # =================================================================================
    # FULL SWEEP (only runs if invoked with --full and preflight above did not exit)
    # =================================================================================
    ckpt_path = 'Paper_Data/sbs_authentic_sweep.npz'
    total_pairs = len(SBS_CONDS)*len(PROTOCOLS)
    pair_i = 0
    for sbscond in SBS_CONDS:
        log(f">>> SBSCOND={sbscond}: building L0 SBS chain over m={M_LIST}")
        chain = build_chain(G_sBs0, sbscond, M_LIST)
        log(f">>> SBSCOND={sbscond}: chain cached")
        for proto in PROTOCOLS:
            pair_i += 1
            scheme = build_scheme(proto)
            D_corr, bx, bp, corr = calibrate_D(scheme, G_sBs0)
            log(f"[{pair_i}/{total_pairs}] sbscond={sbscond} proto={proto}: scheme built "
                f"({len(scheme)} seg), D calibrated (back_x={bx:.4f} back_p={bp:.4f} corr={corr:.4f})")
            new_data = {}
            for readcond in READ_CONDS:
                errs = np.zeros(len(M_LIST)); infids = np.zeros(len(M_LIST))
                for im, m in enumerate(M_LIST):
                    rho_osc_err, ro, qerr = run_scheme_dm(scheme, chain[m], -shift, readcond)
                    ro_corr = D_corr*ro*D_corr.dag()
                    infid = 1 - float(fidelity(G_sBs0_shifted, ro_corr))
                    errs[im] = qerr; infids[im] = infid
                new_data[f'{proto}|{sbscond}|{readcond}|L0|err'] = errs
                new_data[f'{proto}|{sbscond}|{readcond}|L0|infid'] = infids
            d = dict(np.load(ckpt_path)) if os.path.exists(ckpt_path) else {}
            d.update(new_data)
            d['m_list'] = np.array(M_LIST)
            d['gam'] = np.array(gam); d['gamphi'] = np.array(gamphi)
            d['kappa'] = np.array(kappa); d['kappa_phi'] = np.array(kappa_phi)
            np.savez(ckpt_path, **d)
            log(f"[{pair_i}/{total_pairs}] sbscond={sbscond} proto={proto}: checkpoint saved "
                f"({len(d)} keys) elapsed={time.time()-t0:.0f}s")
    log("FULL SWEEP DONE")
