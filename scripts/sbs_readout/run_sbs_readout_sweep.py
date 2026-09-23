"""Fresh codeword -> m rounds of SBS stabilization -> readout.

Sweeps the number of SBS rounds m (M_LIST) on the x-axis and produces, for 7 readout
protocols under a matrix of SBS-noise x readout-noise conditions, both the readout
error P(e) and the post-read oscillator infidelity 1-F.

REUSE (verbatim, per spec):
  - recompute_damage_infid.py: logical(mu), shift, L0/L1/L0raw/STATES/PREF, BB1 table,
    kick(), det_corr(), Mexact(), U(), sc_* scheme builders, build_scheme(), PROTOCOLS,
    readout(), calibrate_D().  ONLY CHANGE: Ncav=200 (was 400).
  - sbs_lib.py: sbs_round(rho_osc, axis, cond, Ncav) for the SBS-round primitive
    (same x=(a+a^dag)/sqrt(2) convention and same logical() comb, Delta=0.34, so the
    two sources are mutually consistent).

SCOPE (per user's mid-task correction): LOGICAL-0 ONLY. STATES={'L0':L0}; no L1 keys
are computed or stored. P(e) for L0 = 1 - Re<pydm|rho_out.ptrace(1)> (no averaging).

Matrix: SBS_CONDS(5) x PROTOCOLS(7) x READ_CONDS(6) x M_LIST(6) x state(1, L0 only).
Checkpointed after each (sbscond, protocol) pair.
"""
import os, sys, time, numpy as np
from qutip import *
import sbs_lib

t0 = time.time()
Ncav = 200          # ONLY deviation from recompute_damage_infid.py (there: 400)
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.
kappa_phi = 1/5000.

aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); py = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(py)
Delta = 0.34; ep = Delta**2; r = -np.log(Delta); sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
shift = np.sqrt(np.pi/2)/2
L0 = (displace(Ncav,-shift)*logical(0)).unit(); L1 = (displace(Ncav,-shift)*logical(1)).unit()
L0raw = logical(0)   # RAW comb (beta=0); used to calibrate D
STATES = {'L0': L0}                       # SCOPE: L0 only (per mid-task correction)
PREF = {'L0': ket2dm(L0), 'L1': ket2dm(L1)}
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} shift={shift:.5f} kappa={kappa} kappa_phi={kappa_phi}", flush=True)

# ---- BB1 / kick / det_corr / U / Mexact -- verbatim from recompute_damage_infid.py ----
BB1 = [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph = BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph = BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def U(K): return (-1j*K).expm(method='dense')

qp = np.load('Paper_Data/qite_det_params.npy')
gp = np.load('Paper_Data/qite_gcrbb1_params.npy')
_blk = np.load('Paper_Data/prepended_block_params.npz')
paramsA = _blk['variantA_params']; paramsB = _blk['variantB_params']

def sc_gcr():
    K,t = det_corr((np.pi/2)*ep/sq, 0+np.pi/2); Kk,tk = kick(3)
    return [('U',U(K),K,t),('U',U(Kk),Kk,tk)]
def sc_bb1():
    L=[]
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_qite():
    L=[]
    for i in range(4):
        K,t=det_corr(qp[2*i],qp[2*i+1]); Kk,tk=kick(i)
        L+=[('U',U(K),K,t),('U',U(Kk),Kk,tk)]
    return L
def sc_gcrbb1():
    L=[]
    Kc,tc=det_corr(gp[0],gp[1]); L.append(('U',U(Kc),Kc,tc))
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_block(p):
    L=[]
    for k in range(4): Kc,tc=det_corr(p[2*k],p[2*k+1]); L.append(('U',U(Kc),Kc,tc))
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_herald():
    L=[]
    for i in range(4):
        Kk,tk=kick(i)
        if i<3: L.append(('M',Mexact(i),None,0.0))
        else:   K,t=det_corr((np.pi/2)*ep/sq,0+np.pi/2); L.append(('U',U(K),K,t))
        L.append(('U',U(Kk),Kk,tk))
    return L

PROTOCOLS = ['GCR','BB1(GCR)-QITE','BB1','GCR-BB1','GCR-BB1-blockA','GCR-BB1-blockB','herald-RUS']
def build_scheme(name):
    return {'GCR':sc_gcr,'BB1(GCR)-QITE':sc_qite,'BB1':sc_bb1,'GCR-BB1':sc_gcrbb1,
            'GCR-BB1-blockA':lambda:sc_block(paramsA),'GCR-BB1-blockB':lambda:sc_block(paramsB),
            'herald-RUS':sc_herald}[name]()

# ---- COPS: recompute_damage_infid.py's readout COPS PLUS the two extras (osc_deph, and nn) ----
sm = tensor(qeye(Ncav),sigmam()); sz = tensor(qeye(Ncav),sigmaz())
aa = tensor(destroy(Ncav),qeye(2)); nn = aa.dag()*aa
COPS = {'noiseless': None,
        'complete': [np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
        'tr_decay': [np.sqrt(gam)*sm], 'tr_deph': [np.sqrt(gamphi/2)*sz],
        'osc_decay': [np.sqrt(kappa)*aa], 'osc_deph': [np.sqrt(kappa_phi)*nn]}
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)
READ_CONDS = ['noiseless','complete','tr_decay','tr_deph','osc_decay','osc_deph']

def readout(scheme, rho_cav, cond):
    rho = tensor(rho_cav, ket2dm(gq)); cops = COPS[cond]
    for kind, op, K, t in scheme:
        if kind == 'M': rho = op*rho*op.dag(); rho = rho/rho.tr()
        elif cops is None: rho = op*rho*op.dag()
        else: rho = mesolve(K/t, rho, [0, t], c_ops=cops, options=opts).states[-1]
    return rho

def calibrate_D(scheme, cond):
    rho = readout(scheme, ket2dm(L0raw), cond); ro = rho.ptrace(0)
    bx = float(np.real(expect(xO, ro)))/np.sqrt(2); bp = float(np.real(expect(pO, ro)))/np.sqrt(2)
    corr = 1j*(bp+np.sqrt(np.pi/2)/2) - bx
    return displace(Ncav, corr)

# ---- FIX (post-preflight, coordinator-approved): sbs_lib.sbs_round's fixed point is the RAW
# (unshifted) logical(mu) comb (see diag_sbs_clean.py, which validates SBS against sb.logical(0,.)
# directly), while the readout circuit above is calibrated for the SHIFTED L0=displace(-shift)*
# logical(0). Feeding the shifted state straight into sbs_round parks it exactly midway between the
# raw-comb L0/L1 fixed points (shift = half the L0/L1 lattice spacing), which is why the first
# pre-flight attempt jumped to ~0.4. The correct recipe (matching run_sbs_readout.py /
# run_sbs_postreadout_validate.py): build/stabilize the chain in the RAW frame (starting from
# L0raw=logical(0)), then apply the -shift displacement ONCE, immediately before readout, at every
# m (including m=0, where this reduces exactly to L0=displace(-shift)*logical(0)).
Dshift = displace(Ncav, -shift)
def to_readout_frame(rho_raw):
    return Dshift*rho_raw*Dshift.dag()

# ---- SBS-round noise conditions (outer axis). sbs_lib.py's COPS_for table doesn't have an
# 'osc_deph' entry (it predates kappa_phi); inject it directly into its cache (same convention:
# nn on the joint (Ncav,2) register, rate kappa_phi=1/5000) rather than editing the library. This
# does not touch any of the validated x/p gate physics -- it only supplies the missing c_ops list
# for the one extra condition this sweep needs. ----
sbs_lib._cops_cache[(Ncav,'osc_deph')] = [np.sqrt(kappa_phi)*nn]
SBS_CONDS = ['complete','tr_decay','tr_deph','osc_decay','osc_deph']
M_LIST = [0,2,4,6,8,10]

def build_chain(state0, sbscond):
    """Return dict m -> RAW-frame (unshifted) oscillator-only density matrix after m SBS rounds
    (1 round = x then p), built incrementally so state at m is reused for m+2. Caller must apply
    to_readout_frame() (the -shift displacement) before feeding a chain state into readout()."""
    chain = {}
    rho = ket2dm(state0)
    chain[0] = rho
    m = 0
    for target in sorted(M_LIST):
        if target == 0:
            continue
        while m < target:
            rho = sbs_lib.sbs_round(rho, 'x', sbscond, Ncav)
            rho = sbs_lib.sbs_round(rho, 'p', sbscond, Ncav)
            m += 1
        chain[m] = rho
    return chain

# =====================================================================================
# MANDATORY PRE-FLIGHT CHECK (Ncav=200, SBSCOND='complete', readcond='noiseless', L0 only)
# =====================================================================================
def preflight():
    print(f"[{time.time()-t0:.0f}s] === PRE-FLIGHT CHECK ===", flush=True)
    sbscond = 'complete'; readcond = 'noiseless'
    chain_L0 = build_chain(L0raw, sbscond)   # RAW frame; shift applied at readout time below
    results = {}
    for proto in ['BB1(GCR)-QITE','GCR','BB1']:
        scheme = build_scheme(proto)
        errs = []
        for m in M_LIST:
            rho_out = readout(scheme, to_readout_frame(chain_L0[m]), readcond)
            p1 = float(np.real(expect(pydm, rho_out.ptrace(1))))
            err = 1 - p1   # L0
            errs.append(err)
        results[proto] = np.array(errs)
        print(f"[{time.time()-t0:.0f}s] preflight {proto:16s} P(e) over m={M_LIST}: "
              f"{np.array2string(np.array(errs), precision=6)}", flush=True)
    return results

if __name__ == '__main__':
    if '--preflight-only' in sys.argv or '--full' not in sys.argv:
        pf = preflight()
        m0_qite = pf['BB1(GCR)-QITE'][0]; m0_gcr = pf['GCR'][0]; m0_bb1 = pf['BB1'][0]
        print(f"\n[{time.time()-t0:.0f}s] SANITY: m=0 P(e) L0 -- "
              f"BB1(GCR)-QITE={m0_qite:.3e} (expect ~1e-5, few e-5) | "
              f"GCR={m0_gcr:.3e} (expect ~5.7e-4) | BB1={m0_bb1:.3e} (expect ~6e-3)", flush=True)
        qite_trace = pf['BB1(GCR)-QITE']
        smooth_ok = bool(np.all(qite_trace < 0.5)) and not (qite_trace[1] > 0.3 if len(qite_trace) > 1 else False)
        print(f"[{time.time()-t0:.0f}s] SANITY: BB1(GCR)-QITE trace stays <0.5, no jump: {smooth_ok}", flush=True)
        if '--preflight-only' in sys.argv:
            sys.exit(0)
        if not (m0_qite < 5e-5 and smooth_ok):
            print(f"[{time.time()-t0:.0f}s] PRE-FLIGHT FAILED -- aborting full sweep.", flush=True)
            sys.exit(1)

    # =================================================================================
    # FULL SWEEP
    # =================================================================================
    ckpt_path = 'Paper_Data/sbs_readout_sweep.npz'
    total_pairs = len(SBS_CONDS)*len(PROTOCOLS)
    pair_i = 0
    for sbscond in SBS_CONDS:
        print(f"[{time.time()-t0:.0f}s] >>> SBSCOND={sbscond}: building L0 SBS chain over m={M_LIST}", flush=True)
        chain = {'L0': build_chain(L0raw, sbscond)}   # RAW frame; shift applied at readout time below
        print(f"[{time.time()-t0:.0f}s] >>> SBSCOND={sbscond}: chain cached", flush=True)
        for proto in PROTOCOLS:
            pair_i += 1
            scheme = build_scheme(proto)
            Dcache = {cond: calibrate_D(scheme, cond) for cond in READ_CONDS}
            print(f"[{time.time()-t0:.0f}s] [{pair_i}/{total_pairs}] sbscond={sbscond} proto={proto}: "
                  f"scheme built ({len(scheme)} seg), D calibrated", flush=True)
            new_data = {}
            for stname in STATES:
                for readcond in READ_CONDS:
                    errs = np.zeros(len(M_LIST)); infids = np.zeros(len(M_LIST))
                    for im, m in enumerate(M_LIST):
                        rho_out = readout(scheme, to_readout_frame(chain[stname][m]), readcond)
                        p1 = float(np.real(expect(pydm, rho_out.ptrace(1))))
                        err = (1-p1) if stname == 'L0' else p1
                        ro_out = rho_out.ptrace(0); D = Dcache[readcond]
                        infid = 1 - float(fidelity(PREF[stname], D*ro_out*D.dag()))
                        errs[im] = err; infids[im] = infid
                    new_data[f'{proto}|{sbscond}|{readcond}|{stname}|err'] = errs
                    new_data[f'{proto}|{sbscond}|{readcond}|{stname}|infid'] = infids
            # checkpoint after each (sbscond, protocol) pair
            d = dict(np.load(ckpt_path)) if os.path.exists(ckpt_path) else {}
            d.update(new_data)
            d['m_list'] = np.array(M_LIST)
            d['gam'] = np.array(gam); d['gamphi'] = np.array(gamphi)
            d['kappa'] = np.array(kappa); d['kappa_phi'] = np.array(kappa_phi)
            np.savez(ckpt_path, **d)
            print(f"[{time.time()-t0:.0f}s] [{pair_i}/{total_pairs}] sbscond={sbscond} proto={proto}: "
                  f"checkpoint saved ({len(d)} keys) elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[{time.time()-t0:.0f}s] DONE -- full sweep complete", flush=True)
