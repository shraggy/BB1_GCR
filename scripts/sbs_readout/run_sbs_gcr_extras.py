"""GCR-only extras, run after the base post-readout-sBs protocol validates.

(a) Repeated GCR: |G_sBs,mu> -D(eps)-> GCR -> sBs(NROUNDS) -> GCR -> sBs(NROUNDS) -> infidelity vs |G_sBs,mu>.
    The two GCR readouts' marginal error probabilities p1,p2 (=qerr from run_scheme_dm, i.e.
    P(wrong outcome) under the py-basis metric) are combined via the standard log-odds
    (Bayesian, independence-assumed) rule:
        L_i = log((1-p_i)/p_i);  L_tot = L1+L2;  p_comb = 1/(1+exp(L_tot))
    reported alongside the two individual qerr values and the final oscillator infidelity.
    Run across all 5 channels, mu in {0,1}, eps sweep.

(b) Heralded-GCR: apply the plain GCR unitary sequence, then PROJECT the ancilla onto |g>
    (Z-basis herald, discard 'e' branch) -- yield = P(g) = Tr[Pg rho]; report the post-selected
    py-basis readout error (conditioned on the g branch) and yield, vs the existing RUS-style
    'heralded' scheme (sc_herald, exact-Kraus M-steps, no discarding -> yield=1 by construction).
    Run across all 5 channels, mu in {0,1}, eps sweep.
"""
import sys, time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
Ncav = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
CHANNELS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['noiseless','complete','tr_decay','tr_deph','osc_decay']
PARTS = sys.argv[5].split(',') if len(sys.argv) > 5 else ['a','b']
Mmax = 20
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.

def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

# ================= sBs (i=2 convention) =================
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

def grid(state, qubit):
    measState = tensor(qubit, state)
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=[], options=Options(nsteps=5000)).states[-1]
    measState = Rx*measState*Rx1
    measState = mesolve(Hamp, measState, [0, np.sqrt(2)*np.cosh(Delta**2)], c_ops=[], options=Options(nsteps=5000)).states[-1]
    measState = Rx1*measState*Rx
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=[], options=Options(nsteps=10000)).states[-1]
    return measState.ptrace(1).unit(), measState.ptrace(0).unit()

def applyMeasurement(state, correctionDirection):
    displacePauli = displace(Ncav, a_sbs/4/np.sqrt(2)*np.sinh(DeltaEpsilon**2)+1j*a_sbs/4/np.sqrt(2)*np.cosh(DeltaEpsilon**2))
    displacePaulid = displacePauli.dag()
    M = 0.5*(displacePauli+displacePaulid)
    M = displacePauli*M if correctionDirection == 1 else displacePaulid*M
    newState = M*state*M.dag()
    return newState/newState.tr()

def computeTrajCooling(tempState, Mmax):
    stateList = [tempState]
    for i in range(Mmax):
        tempState, tempqubit = grid(tempState, ket2dm(px))
        tempState1, tempqubit = grid(Urot*tempState*Urot1, ket2dm(px))
        tempState = Urot1*tempState1*Urot
        if i == measIndex:
            tempState = applyMeasurement(tempState, correctionDirection=1)
            tempState = applyMeasurement(tempState, correctionDirection=-1)
        stateList.append(tempState)
    return stateList

def sbs_round(tempState):
    tempState, _ = grid(tempState, ket2dm(px))
    tempState1, _ = grid(Urot*tempState*Urot1, ket2dm(px))
    tempState = Urot1*tempState1*Urot
    return tempState

log(f"Ncav={Ncav} building G_sBs via computeTrajCooling(Mmax={Mmax})...")
stateList = computeTrajCooling(ket2dm(basis(Ncav,0)), Mmax)
G_SBS = {1: stateList[-1], 0: stateList[-2]}
log("G_sBs built.")

# ================= readout scheme (i=sqrt(2) convention) =================
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); pyq = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(pyq)
Pg = tensor(qeye(Ncav), ket2dm(basis(2,0)))
ep = Delta**2; sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
u = np.sqrt(np.pi)/(2*np.sqrt(2)); shift = np.sqrt(np.pi/2)/2
qp = np.load('Paper_Data/qite_det_params.npy')
BB1 = [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph = BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph = BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def Uu(K): return (-1j*K).expm(method='dense')
def sc_gcr():
    K,t = det_corr((np.pi/2)*ep/sq, np.pi/2); Kk,tk = kick(3)
    return [('U',Uu(K),K,t),('U',Uu(Kk),Kk,tk)]
def sc_herald():
    L = []
    for i in range(4):
        Kk,tk = kick(i)
        if i < 3: L.append(('M',Mexact(i),None,0.0))
        else: K,t = det_corr((np.pi/2)*ep/sq,np.pi/2); L.append(('U',Uu(K),K,t))
        L.append(('U',Uu(Kk),Kk,tk))
    return L
sm = tensor(qeye(Ncav),sigmam()); sz = tensor(qeye(Ncav),sigmaz()); aa = tensor(aC,qeye(2))
COPS = {
    'noiseless': None,
    'complete':  [np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
    'tr_decay':  [np.sqrt(gam)*sm],
    'tr_deph':   [np.sqrt(gamphi/2)*sz],
    'osc_decay': [np.sqrt(kappa)*aa],
}
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

def apply_scheme_full(scheme, rho, cops):
    """Apply the scheme's unitary/Kraus steps to a full (osc x qubit) density matrix; no herald."""
    for kind,op,K,t in scheme:
        if kind == 'M':
            rho = op*rho*op.dag(); rho = rho/rho.tr()
        elif cops is None:
            rho = op*rho*op.dag()
        else:
            rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    return rho

def qerr_of(rho_full):
    return 1 - float(np.real(expect(pydm, rho_full.ptrace(1))))

def gcr_readout(rho_osc, cond):
    """One GCR shot on an oscillator dm; returns (qerr, rho_osc_after)."""
    rho = tensor(rho_osc, ket2dm(gq))
    rho = apply_scheme_full(sc_gcr(), rho, COPS[cond])
    return qerr_of(rho), rho.ptrace(0)

def do_sbs(rho_osc, nrounds):
    for _ in range(nrounds):
        rho_osc = sbs_round(rho_osc)
    return rho_osc

epsA = np.linspace(0, u, NE)
results = {'epsA': epsA, 'u': u, 'NROUNDS': NROUNDS}

if 'a' in PARTS:
    log("=== Extra (a): repeated GCR, log-odds combination ===")
    for cond in CHANNELS:
        for mu in [0,1]:
            psi_target = G_SBS[mu]
            p1s=[]; p2s=[]; pcombs=[]; inf_finals=[]
            for eps in epsA:
                beta = eps - shift
                rho_err = displace(Ncav,beta)*psi_target*displace(Ncav,beta).dag()
                q1, ro1 = gcr_readout(rho_err, cond)
                ro1s = do_sbs(ro1, NROUNDS)
                q2, ro2 = gcr_readout(ro1s, cond)
                ro2s = do_sbs(ro2, NROUNDS)
                inf_final = 1 - float(fidelity(ro2s, psi_target))
                p1c = min(max(q1,1e-12),1-1e-12); p2c = min(max(q2,1e-12),1-1e-12)
                L = np.log((1-p1c)/p1c) + np.log((1-p2c)/p2c)
                pcomb = 1/(1+np.exp(L))
                p1s.append(q1); p2s.append(q2); pcombs.append(pcomb); inf_finals.append(inf_final)
                log(f"(a) {cond:10s} mu={mu} eps={eps:.3f}  q1={q1:.4e} q2={q2:.4e} p_comb={pcomb:.4e}  inf_final={inf_final:.4e}")
            key=f"a|{cond}|mu{mu}"
            results[f"{key}|q1"]=np.array(p1s); results[f"{key}|q2"]=np.array(p2s)
            results[f"{key}|pcomb"]=np.array(pcombs); results[f"{key}|inf"]=np.array(inf_finals)
            np.savez('Paper_Data/sbs_gcr_extras.npz', **results)

if 'b' in PARTS:
    log("=== Extra (b): heralded-GCR (post-select ancilla='g') vs RUS-of-BB1(GCR) ===")
    for cond in CHANNELS:
        for mu in [0,1]:
            psi_target = G_SBS[mu]
            yields=[]; qerr_hs=[]; qerr_rus=[]
            for eps in epsA:
                beta = eps - shift
                rho_err = displace(Ncav,beta)*psi_target*displace(Ncav,beta).dag()
                # heralded-GCR: apply GCR sequence, project ancilla onto |g>, renormalize
                rho = tensor(rho_err, ket2dm(gq))
                rho = apply_scheme_full(sc_gcr(), rho, COPS[cond])
                yield_g = float(np.real((Pg*rho).tr()))
                rho_g = (Pg*rho*Pg)/yield_g
                qerr_h = qerr_of(rho_g)
                # RUS-of-BB1(GCR) (existing heralded scheme, exact-Kraus, no discard -> yield=1)
                rho2 = tensor(rho_err, ket2dm(gq))
                rho2 = apply_scheme_full(sc_herald(), rho2, COPS[cond])
                qerr_r = qerr_of(rho2)
                yields.append(yield_g); qerr_hs.append(qerr_h); qerr_rus.append(qerr_r)
                log(f"(b) {cond:10s} mu={mu} eps={eps:.3f}  herald_yield={yield_g:.4f} herald_qerr={qerr_h:.4e}  RUS_qerr(yield=1)={qerr_r:.4e}")
            key=f"b|{cond}|mu{mu}"
            results[f"{key}|yield"]=np.array(yields); results[f"{key}|qerr_h"]=np.array(qerr_hs); results[f"{key}|qerr_rus"]=np.array(qerr_rus)
            np.savez('Paper_Data/sbs_gcr_extras.npz', **results)

log("EXTRAS DONE")
