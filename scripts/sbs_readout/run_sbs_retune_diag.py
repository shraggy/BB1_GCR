"""Diagnostic: is the GCR readout's ~8% error on the binomial |G_sBs,0> a tuning/calibration
mismatch, or a genuine limit of the state?

Method: after applying the FIXED GCR circuit (unchanged) to |G_sBs,0> (at nominal eps=0, i.e. the
scheme's own hard-coded input offset beta0=-shift), take the reduced ANCILLA qubit density matrix
rho_q = rho.ptrace(1). For ANY qubit density matrix, the best possible binary-readout error over
ALL choices of projective measurement axis (not just the scheme's fixed py-basis
(|0>+i|1>)/sqrt(2)) is analytically:
    qerr_min = (1 - |r|) / 2
where r=(<sx>,<sy>,<sz>) is the Bloch vector of rho_q. This is the "optimal readout target angle"
scan collapsed to closed form (scanning all Bloch-sphere axes numerically would give the same
answer up to numerical precision -- we also do a brute numeric scan as a cross check).

Separately, scan the INPUT DISPLACEMENT OFFSET (the beta the scheme's own preceding displacement
applies before the GCR pulses) over a small 2D (real,imag) grid around the nominal beta0=-shift,
to see whether re-centering the input state (rather than re-choosing the readout basis) recovers
a higher |r| / lower error. Report both scans, plus a joint (best db, best basis) result.

If qerr_min stays ~8%, the binomial-sBs state genuinely gives the transmon ancilla a mixed/impure
post-GCR state (real limit). If qerr_min drops toward ~1e-3, the ORIGINAL fixed py-basis (and/or
fixed input offset) was simply mismatched to this state's phase-space centering -- a tuning bug,
not a physical limit.
"""
import sys, time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
Ncav = int(sys.argv[1]) if len(sys.argv) > 1 else 100
Mmax = 20

def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

# ================= sBs (i=2 convention) -- build G_sBs0 =================
g = basis(2,0); e = basis(2,1); px = (g+e).unit()
aOp = destroy(Ncav); aOp1 = aOp.dag()
xOp_s = (aOp+aOp1)/2; pOp_s = (-1j)*(aOp-aOp1)/2

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

log(f"Ncav={Ncav} building G_sBs0 via computeTrajCooling(Mmax={Mmax})...")
stateList = computeTrajCooling(ket2dm(basis(Ncav,0)), Mmax)
G_sBs0 = stateList[-2]
log("G_sBs0 built.")

# ================= readout scheme (i=sqrt(2) convention) =================
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0)
ep = Delta**2; sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
u = np.sqrt(np.pi)/(2*np.sqrt(2)); shift = np.sqrt(np.pi/2)/2
BB1 = [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph = BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Uu(K): return (-1j*K).expm(method='dense')
def sc_gcr():
    K,t = det_corr((np.pi/2)*ep/sq, np.pi/2); Kk,tk = kick(3)
    return [('U',Uu(K),K,t),('U',Uu(Kk),Kk,tk)]

def apply_gcr_noiseless(rho_osc):
    rho = tensor(rho_osc, ket2dm(gq))
    for kind,op,K,t in sc_gcr():
        rho = op*rho*op.dag()
    return rho

def bloch_vector(rho_q):
    rx_ = float(np.real(expect(sigmax(), rho_q)))
    ry_ = float(np.real(expect(sigmay(), rho_q)))
    rz_ = float(np.real(expect(sigmaz(), rho_q)))
    return np.array([rx_,ry_,rz_])

pydm = ket2dm((basis(2,0)+1j*basis(2,1)).unit())
def qerr_pybasis(rho_q): return 1 - float(np.real(expect(pydm, rho_q)))

# ---- baseline: nominal beta0 = -shift (eps=0), fixed py-basis ----
beta0 = 0.0 - shift
rho_final0 = apply_gcr_noiseless(displace(Ncav,beta0)*G_sBs0*displace(Ncav,beta0).dag())
rho_q0 = rho_final0.ptrace(1)
qerr_current = qerr_pybasis(rho_q0)
r0 = bloch_vector(rho_q0); rmag0 = np.linalg.norm(r0)
qerr_bloch_opt0 = (1-rmag0)/2
log(f"BASELINE (beta0={beta0:.4f}, fixed py-basis): qerr_current={qerr_current:.4e}")
log(f"BASELINE Bloch vector r=({r0[0]:.4f},{r0[1]:.4f},{r0[2]:.4f})  |r|={rmag0:.4f}  qerr_Bloch_optimal={qerr_bloch_opt0:.4e}")

# brute-force numeric cross-check: scan measurement axis over full Bloch sphere
best_numeric = 1.0; best_ang = None
for theta in np.linspace(0, np.pi, 19):        # colatitude
    for phi in np.linspace(0, 2*np.pi, 37):    # azimuth
        n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        proj = ket2dm(( (np.cos(theta/2))*basis(2,0) + np.exp(1j*phi)*np.sin(theta/2)*basis(2,1) ).unit())
        qe = 1 - float(np.real(expect(proj, rho_q0)))
        if qe < best_numeric: best_numeric = qe; best_ang = (theta,phi)
log(f"Numeric cross-check best qerr over Bloch-sphere scan = {best_numeric:.4e} at (theta,phi)={best_ang} (should match qerr_Bloch_optimal)")

# ---- scan input displacement offset db around beta0, re-derive Bloch-optimal qerr for each ----
log("Scanning input displacement offset db (real,imag) around beta0=-shift...")
dbr = np.linspace(-0.4, 0.4, 9)
dbi = np.linspace(-0.4, 0.4, 9)
best = {'rmag': rmag0, 'db': 0+0j, 'qerr_opt': qerr_bloch_opt0}
grid_results = np.zeros((len(dbr), len(dbi)))
for i, drr in enumerate(dbr):
    for j, dii in enumerate(dbi):
        db = drr + 1j*dii
        beta = beta0 + db
        rho_f = apply_gcr_noiseless(displace(Ncav,beta)*G_sBs0*displace(Ncav,beta).dag())
        rq = rho_f.ptrace(1)
        rvec = bloch_vector(rq); rmag = np.linalg.norm(rvec)
        grid_results[i,j] = rmag
        if rmag > best['rmag']:
            best = {'rmag': rmag, 'db': db, 'qerr_opt': (1-rmag)/2}
log(f"Best over db-scan: db={best['db']:.4f}  |r|={best['rmag']:.4f}  qerr_Bloch_optimal={best['qerr_opt']:.4e}")
log(f"(for reference, |r| range over scan: min={grid_results.min():.4f} max={grid_results.max():.4f})")

np.savez('Paper_Data/sbs_retune_diag.npz',
         qerr_current=qerr_current, r0=r0, rmag0=rmag0, qerr_bloch_opt0=qerr_bloch_opt0,
         best_numeric=best_numeric, best_ang=best_ang,
         dbr=dbr, dbi=dbi, grid_results=grid_results,
         best_db=best['db'], best_rmag=best['rmag'], best_qerr_opt=best['qerr_opt'])
log("RETUNE DIAG DONE")
