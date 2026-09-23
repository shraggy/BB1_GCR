"""Readout-on-sBs-prepared-state test. Builds |G_sBs,mu> via the authentic sBs
(computeTrajCooling, verbatim port from Initial_repo.ipynb cell 41 -- see run_authentic_sbs.py),
then feeds it as the INPUT to the GCR-family readout (run_mesolve_4.py's scheme/COPS/xO,pO
conventions -- i=sqrt(2), convention-independent from the sBs's own i=2 internal convention since
displace()/squeeze() never reference xOp/pOp). For each scheme x noise condition:
  - sweeps a displacement error eps across the logical-0 window [0,u] (beta=eps-shift, applied via
    D(beta)*G_sBs0*D(beta).dag() -- same recipe as run_corr_disp.py, just with G_sBs0 in place of
    the ideal Gaussian logical(0))
  - reports readout error qerr=1-P(+1)
  - calibrates ONE corrective displacement D_corr on the no-error (beta=0) NOISELESS readout of
    G_sBs0 (cell-14 recipe), then reports oscillator fidelity of the readout output (raw and
    D_corr-corrected) RELATIVE TO the errored G_sBs input (D(beta)*G_sBs0*D(beta).dag()), i.e. does
    the readout preserve the sBs-prepared codeword, not the ideal one.
SCHEMES arg (sys.argv[3], comma-separated) selects which of GCR,BB1(GCR),BB1,heralded to run
(default: GCR only).
"""
import sys, time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
Ncav = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SCHEME_NAMES = sys.argv[3].split(',') if len(sys.argv) > 3 else ['GCR']
Mmax = 20
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.

# ================= Part 1: build G_sBs via authentic sBs (i=2 convention, local) =================
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

print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} building G_sBs via computeTrajCooling(Mmax={Mmax})...", flush=True)
stateList = computeTrajCooling(ket2dm(basis(Ncav,0)), Mmax)
G_sBs1 = stateList[-1]     # mu=1-like (Mmax rounds)
G_sBs0 = stateList[-2]     # mu=0-like (Mmax-1 rounds)
state_0 = ket2dm(logical1(0,Delta,N,Ncav)); state_1 = ket2dm(logical1(1,Delta,N,Ncav))
print(f"[{time.time()-t0:.0f}s] G_sBs0 fid to binomial-L0={fidelity(G_sBs0,state_0):.4f}  "
      f"G_sBs1 fid to binomial-L1={fidelity(G_sBs1,state_1):.4f}", flush=True)

# ================= Part 2: readout scheme (i=sqrt(2) convention, independent xO,pO) =================
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); pyq = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(pyq)
ep = Delta**2; r = -np.log(Delta); sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
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
COPS = {'noiseless':None, 'complete':[np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa]}
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
    return rho_osc_err, rho.ptrace(0), qerr

def calibrate_D(scheme, rho0_in):
    rho_osc_err0, ro0, _ = run_scheme_dm(scheme, rho0_in, 0.0, 'noiseless')
    back_x = float(np.real(expect(xO, ro0)))/np.sqrt(2)
    back_p = float(np.real(expect(pO, ro0)))/np.sqrt(2)
    corr = 1j*(back_p + np.sqrt(np.pi/2)/2) - back_x
    return displace(Ncav, corr), back_x, back_p, corr

epsA = np.linspace(0, u, NE)
results = {}
for scn in SCHEME_NAMES:
    scheme = SCHEMES[scn]()
    D, bx, bp, corr = calibrate_D(scheme, G_sBs0)
    print(f"[{time.time()-t0:.0f}s] {scn}: back_x={bx:.4f} back_p={bp:.4f} corr={corr:.4f}", flush=True)
    for cond in ['noiseless','complete']:
        qerrs = []; raw_fids = []; corr_fids = []
        for eps in epsA:
            beta = eps - shift
            rho_osc_err, ro, qerr = run_scheme_dm(scheme, G_sBs0, beta, cond)
            f_raw = float(fidelity(rho_osc_err, ro))
            ro_corr = D*ro*D.dag()
            f_corr = float(fidelity(rho_osc_err, ro_corr))
            qerrs.append(qerr); raw_fids.append(f_raw); corr_fids.append(f_corr)
        qerrs = np.array(qerrs); raw_fids = np.array(raw_fids); corr_fids = np.array(corr_fids)
        results[f'{scn}|{cond}|qerr'] = qerrs
        results[f'{scn}|{cond}|raw_fid'] = raw_fids
        results[f'{scn}|{cond}|corr_fid'] = corr_fids
        print(f"[{time.time()-t0:.0f}s] {scn:10s} {cond:10s} "
              f"qerr@peak={qerrs[0]:.4e} qerr@u={qerrs[-1]:.4e}  "
              f"raw_fid@peak={raw_fids[0]:.4f} corr_fid@peak={corr_fids[0]:.4f} corr_fid@u={corr_fids[-1]:.4f}",
              flush=True)
        np.savez('Paper_Data/sbs_readout_fidelity.npz', epsA=epsA, u=u, **results)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
