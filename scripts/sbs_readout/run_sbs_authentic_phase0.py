"""PHASE 0 stability check for the AUTHENTIC SBS primitive (grid()/Hamx/Hamp mesolve, i=2
convention), verbatim from run_sbs_readout.py (computeTrajCooling, Mmax=20) and
run_sbs_full_sweep.py (standalone sbs_round(tempState), lines ~77-81). Ncav=200.

Recipe: build the noiseless-cooled codeword G_sBs0 (same recipe as run_sbs_readout.py lines
66-81), then apply sbs_round() NOISELESSLY 10 times starting FROM G_sBs0 itself, printing
per-round fidelity F(rho_k, G_sBs0) and ancilla-free purity Tr(rho_k^2). PASS if F stays
~1 (>0.95) and does not collapse/oscillate the way the broken sbs_lib.sbs_round did.
"""
import time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
Ncav = 200
Mmax = 20
Delta = 0.34

g = basis(2,0); e = basis(2,1); px = (g+e).unit()
aOp = destroy(Ncav); aOp1 = aOp.dag()
xOp_s = (aOp+aOp1)/2; pOp_s = (-1j)*(aOp-aOp1)/2

def logical1(mu, delta, N, Ncav, normalize=True):
    psi = 0*basis(Ncav); r = -np.log(delta); a = np.sqrt(np.pi/2)
    for n in range(-int((N+mu)/2), int((N+mu)/2)-mu+1):
        psi = psi + np.sqrt(sp.special.comb(N, n+mu+int(N/2)))*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    if normalize: psi = psi/psi.norm()
    return psi

N = int(np.floor(0.32/Delta**2)); measIndex = Mmax-4
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
    """standalone SBS round, verbatim from run_sbs_full_sweep.py lines 77-81"""
    tempState, _ = grid(tempState, ket2dm(px))
    tempState1, _ = grid(Urot*tempState*Urot1, ket2dm(px))
    tempState = Urot1*tempState1*Urot
    return tempState

print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} building G_sBs0 via computeTrajCooling(Mmax={Mmax})...", flush=True)
stateList = computeTrajCooling(ket2dm(basis(Ncav,0)), Mmax)
G_sBs1 = stateList[-1]
G_sBs0 = stateList[-2]
state_0 = ket2dm(logical1(0,Delta,N,Ncav)); state_1 = ket2dm(logical1(1,Delta,N,Ncav))
print(f"[{time.time()-t0:.0f}s] G_sBs0 fid to binomial-L0={fidelity(G_sBs0,state_0):.4f}  "
      f"G_sBs1 fid to binomial-L1={fidelity(G_sBs1,state_1):.4f}", flush=True)

print(f"[{time.time()-t0:.0f}s] === PHASE 0: 14 NOISELESS sbs_round() iterations starting FROM G_sBs0 ===", flush=True)
rho_k = G_sBs0
f0 = float(fidelity(rho_k, G_sBs0)); f1_0 = float(fidelity(rho_k, G_sBs1)); p0 = float(np.real((rho_k*rho_k).tr()))
print(f"round  0: F(G_sBs0)={f0:.6f}  F(G_sBs1)={f1_0:.6f}  purity={p0:.6f}  [{time.time()-t0:.0f}s]", flush=True)
for k in range(1, 15):
    rho_k = sbs_round(rho_k)
    fk = float(fidelity(rho_k, G_sBs0)); fk1 = float(fidelity(rho_k, G_sBs1)); pk = float(np.real((rho_k*rho_k).tr()))
    print(f"round {k:2d}: F(G_sBs0)={fk:.6f}  F(G_sBs1)={fk1:.6f}  purity={pk:.6f}  [{time.time()-t0:.0f}s]", flush=True)
print(f"[{time.time()-t0:.0f}s] PHASE 0 DONE", flush=True)
