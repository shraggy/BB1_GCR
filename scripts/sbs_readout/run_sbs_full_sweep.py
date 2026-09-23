"""Full sweep of the corrected post-readout-sBs protocol:
  1. |G_sBs,mu> = computeTrajCooling output (authentic sBs, cell-41 grid()/i=2 convention).
  2. rho = D(beta)|G_sBs,mu>D(beta).dag(), beta=eps-shift (displacement-error injection).
  3. apply readout protocol (ancilla qubit, noise channel) -> qerr; trace out ancilla.
  4. apply NROUNDS post-readout sBs rounds (grid() double-round, no measurement, always noiseless).
  5. oscillator infidelity = 1 - fidelity(final oscillator, |G_sBs,mu>).
Sweeps: protocol in {GCR, BB1(GCR), BB1, heralded} x channel in {noiseless, complete, tr_decay,
tr_deph, osc_decay} x mu in {0,1} x eps in linspace(0,u,NE).
Saves incrementally to Paper_Data/sbs_full_sweep.npz; also prints a running summary line per case.
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
PROTOCOLS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['GCR','BB1(GCR)','BB1','heralded']
CHANNELS = sys.argv[5].split(',') if len(sys.argv) > 5 else ['noiseless','complete','tr_decay','tr_deph','osc_decay']
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
G_sBs1 = stateList[-1]
G_sBs0 = stateList[-2]
state_0 = ket2dm(logical1(0,Delta,N,Ncav)); state_1 = ket2dm(logical1(1,Delta,N,Ncav))
log(f"G_sBs0 fid to binomial-L0={fidelity(G_sBs0,state_0):.4f}  G_sBs1 fid to binomial-L1={fidelity(G_sBs1,state_1):.4f}")
G_SBS = {0: G_sBs0, 1: G_sBs1}

# ================= readout scheme (i=sqrt(2) convention) =================
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); pyq = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(pyq)
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
SCHEMES = {'GCR':sc_gcr,'BB1(GCR)':sc_qite,'BB1':sc_bb1,'heralded':sc_herald}
sm = tensor(qeye(Ncav),sigmam()); sz = tensor(qeye(Ncav),sigmaz()); aa = tensor(aC,qeye(2))
COPS = {
    'noiseless': None,
    'complete':  [np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
    'tr_decay':  [np.sqrt(gam)*sm],
    'tr_deph':   [np.sqrt(gamphi/2)*sz],
    'osc_decay': [np.sqrt(kappa)*aa],
}
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

def run_scheme_dm(scheme, rho_in, beta, cond):
    D = displace(Ncav, beta)
    rho_osc_err = D*rho_in*D.dag()
    rho = tensor(rho_osc_err, ket2dm(gq)); cops = COPS[cond]
    for kind,op,K,t in scheme:
        if kind == 'M':
            rho = op*rho*op.dag(); rho = rho/rho.tr()
        elif cops is None:
            rho = op*rho*op.dag()
        else:
            rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    qerr = 1 - float(np.real(expect(pydm, rho.ptrace(1))))
    return rho.ptrace(0), qerr

epsA = np.linspace(0, u, NE)
results = {'epsA': epsA, 'u': u, 'NROUNDS': NROUNDS}
ncases = len(PROTOCOLS)*len(CHANNELS)*2*NE
icase = 0
for proto in PROTOCOLS:
    for cond in CHANNELS:
        for mu in [0,1]:
            qerrs = []; inf_final = []
            psi_target = G_SBS[mu]
            scheme = SCHEMES[proto]()
            for eps in epsA:
                icase += 1
                beta = eps - shift
                rho_after, qerr = run_scheme_dm(scheme, psi_target, beta, cond)
                rho_k = rho_after
                for k in range(NROUNDS):
                    rho_k = sbs_round(rho_k)
                inf_k = 1 - float(fidelity(rho_k, psi_target))
                qerrs.append(qerr); inf_final.append(inf_k)
                log(f"[{icase}/{ncases}] {proto:10s} {cond:10s} mu={mu} eps={eps:.3f}  qerr={qerr:.4e}  inf@{NROUNDS}rounds={inf_k:.4e}")
            key = f"{proto}|{cond}|mu{mu}"
            results[f"{key}|qerr"] = np.array(qerrs)
            results[f"{key}|inf"] = np.array(inf_final)
            np.savez('Paper_Data/sbs_full_sweep.npz', **results)
log("FULL SWEEP DONE")
