"""Diagnostic: is herald-RUS's P(e)~0.53 at beta=-shift a mis-centering artifact of its
non-unitary momentum-filter ('M'-kind Mexact) segments, which are not shift-covariant like the
six unitary protocols?

Reuses the EXACT definitions from run_sbs_authentic_sweep.py:
  - Part 1 (grid/applyMeasurement/computeTrajCooling) to rebuild G_sBs0 = stateList[-2] via the
    same NOISELESS 20-round cooling, Ncav=200.
  - Part 2 (i=sqrt(2) readout convention: kick/det_corr/Mexact/Uu/sc_gcr/sc_qite/sc_herald,
    run_scheme_dm) verbatim.

Only NOISELESS readout (cond='noiseless') is used throughout, and qerr = P(e) is read directly
off run_scheme_dm's return (qubit expect(pydm, rho.ptrace(1))) -- independent of D/calibrate_D,
so infidelity/D-correction is skipped entirely (not needed for this diagnostic).

Does NOT modify run_sbs_authentic_sweep.py or any .npz.
"""
import time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

Ncav = 200
Mmax = 20

# ================= Part 1: authentic sBs, i=2 convention (verbatim, run_sbs_authentic_sweep.py) =================
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

log(f"Ncav={Ncav} building G_sBs0 (m=0 input) via NOISELESS computeTrajCooling(Mmax={Mmax})...")
stateList = computeTrajCooling(ket2dm(basis(Ncav, 0)), Mmax)
G_sBs1 = stateList[-1]
G_sBs0 = stateList[-2]
state_0 = ket2dm(logical1(0, Delta, N, Ncav)); state_1 = ket2dm(logical1(1, Delta, N, Ncav))
log(f"G_sBs0 fid to binomial-L0={fidelity(G_sBs0,state_0):.4f}  G_sBs1 fid to binomial-L1={fidelity(G_sBs1,state_1):.4f}")

# ================= Part 2: readout scheme, i=sqrt(2) convention (verbatim) =================
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

COPS_READOUT = {'noiseless': None}
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

def run_scheme_dm(scheme, rho_in, beta, cond):
    """Verbatim from run_sbs_authentic_sweep.py."""
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

def qerr_at(scheme, beta):
    _, _, qerr = run_scheme_dm(scheme, G_sBs0, beta, 'noiseless')
    return qerr

# ================= Diagnostic scan =================
scheme_herald = sc_herald()
scheme_gcr = sc_gcr()
scheme_qite = sc_qite()

log("Computing reference points (beta=0.0, beta=-shift) for herald-RUS, GCR, BB1(GCR)-QITE...")
herald_at0 = qerr_at(scheme_herald, 0.0)
herald_at_negshift = qerr_at(scheme_herald, -shift)
gcr_at0 = qerr_at(scheme_gcr, 0.0)
gcr_at_negshift = qerr_at(scheme_gcr, -shift)
qite_at0 = qerr_at(scheme_qite, 0.0)
qite_at_negshift = qerr_at(scheme_qite, -shift)

log(f"herald-RUS   P(e) @ beta=0.0     = {herald_at0:.4f}")
log(f"herald-RUS   P(e) @ beta=-shift  = {herald_at_negshift:.4f}")
log(f"GCR          P(e) @ beta=0.0     = {gcr_at0:.4f}")
log(f"GCR          P(e) @ beta=-shift  = {gcr_at_negshift:.4f}")
log(f"BB1(GCR)-QITE P(e) @ beta=0.0    = {qite_at0:.4f}")
log(f"BB1(GCR)-QITE P(e) @ beta=-shift = {qite_at_negshift:.4f}")

n_steps = 33
beta_vals = np.linspace(-2*shift, 2*shift, n_steps)

log(f"Scanning herald-RUS over REAL beta in [-2*shift, 2*shift] ({n_steps} steps)...")
qerr_real = np.zeros(n_steps)
for i, b in enumerate(beta_vals):
    qerr_real[i] = qerr_at(scheme_herald, b)
    log(f"  [real {i+1}/{n_steps}] beta={b:+.4f}  P(e)={qerr_real[i]:.4f}")

log(f"Scanning herald-RUS over IMAG beta = 1j*b, b in [-2*shift, 2*shift] ({n_steps} steps)...")
qerr_imag = np.zeros(n_steps)
for i, b in enumerate(beta_vals):
    qerr_imag[i] = qerr_at(scheme_herald, 1j*b)
    log(f"  [imag {i+1}/{n_steps}] beta={1j*b}  P(e)={qerr_imag[i]:.4f}")

imin_real = int(np.argmin(qerr_real))
imin_imag = int(np.argmin(qerr_imag))

print("\n" + "="*78)
print("SUMMARY TABLE")
print("="*78)
print(f"shift = sqrt(pi/2)/2 = {shift:.6f}")
print()
print("Reference points (noiseless readout, G_sBs0, no D-correction):")
print(f"  {'protocol':16s}  {'beta=0.0':>10s}  {'beta=-shift':>12s}")
print(f"  {'herald-RUS':16s}  {herald_at0:10.4f}  {herald_at_negshift:12.4f}")
print(f"  {'GCR':16s}  {gcr_at0:10.4f}  {gcr_at_negshift:12.4f}")
print(f"  {'BB1(GCR)-QITE':16s}  {qite_at0:10.4f}  {qite_at_negshift:12.4f}")
print()
print(f"herald-RUS REAL-beta scan minimum:  P(e)={qerr_real[imin_real]:.4f}  at beta={beta_vals[imin_real]:+.4f}"
      f"  ({beta_vals[imin_real]/shift:+.3f} * shift)")
print(f"herald-RUS IMAG-beta scan minimum:  P(e)={qerr_imag[imin_imag]:.4f}  at beta={1j*beta_vals[imin_imag]}"
      f"  (b/shift={beta_vals[imin_imag]/shift:+.3f})")
print()
print("Full REAL-beta scan (beta, P(e)):")
for b, q in zip(np.round(beta_vals,4), np.round(qerr_real,4)):
    print(f"    beta={b:+.4f}   P(e)={q:.4f}")
print()
print("Full IMAG-beta scan (b where beta=1j*b, P(e)):")
for b, q in zip(np.round(beta_vals,4), np.round(qerr_imag,4)):
    print(f"    b={b:+.4f}   P(e)={q:.4f}")
print("="*78)
log("DONE")
