"""Authentic sBs port, verbatim from Initial_repo.ipynb cell 41 (+ logical1 from cell 36, qubit
defs from cell 1, oscillator-operator convention i=2 active at cell 41 via cell 33's redefinition:
  xOp=(aOp+aOp1)/2, pOp=(-1j)*(aOp-aOp1)/2   (NOT the /sqrt(2) convention used elsewhere in this repo)
Tensor order in grid(): (QUBIT, OSC) -- qubit index 0, osc index 1.

grid() IS the sBs stabilization (small-Hamx, Rx-conjugate, big-Hamp, Rx1-conjugate, small-Hamx,
trace qubit) -- this is what should "cool" an arbitrary oscillator state toward the binomial GKP
steady state when iterated (with the U/U1 pi/2-rotation conjugation to alternate quadratures).
applyMeasurement is a SEPARATE one-shot logical-Pauli projection (used once at measIndex inside
computeTrajCooling to pick a definite logical value), NOT part of the stabilization itself.

Reports, in order:
  (A) grid()-only cooling trace: start from vacuum |0>, iterate grid()+U-conjugation for Mmax
      rounds (no applyMeasurement), track fidelity to the binomial-GKP logical1(0)/logical1(1)
      every round -- this is the real "does sBs cool toward the GKP codeword" validation.
  (B) full computeTrajCooling (grid rounds + the one applyMeasurement pair at measIndex=Mmax-4):
      report fidelity of the final two states (Mmax, Mmax-1 rounds) against logical1(1)/logical1(0)
      respectively (mirrors cell 41's own comparison, which pairs across the round-parity flip).
"""
import sys, time, numpy as np
import scipy as sp
from scipy import special
from qutip import *
from qutip.qip.operations import rx

t0 = time.time()
Ncav = int(sys.argv[1]) if len(sys.argv) > 1 else 100
Mmax = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# --- qubit states (cell 1) ---
g = basis(2,0); e = basis(2,1)
px = (g+e).unit(); mx = (g-e).unit(); py = (g+1j*e).unit(); my = (g-1j*e).unit()

# --- oscillator operators, i=2 convention active at cell 41 (set in cell 33) ---
aOp = destroy(Ncav); aOp1 = aOp.dag()
ii = 2
xOp = (aOp+aOp1)/ii
pOp = (-1j)*(aOp-aOp1)/ii

# --- logical1: binomial GKP (cell 36) ---
def logical1(mu, delta, N, Ncav, normalize=True):
    psi = 0*basis(Ncav); r = -np.log(delta); a = np.sqrt(np.pi/2)
    for n in range(-int((N+mu)/2), int((N+mu)/2)-mu+1):
        psi = psi + np.sqrt(sp.special.comb(N, n+mu+int(N/2)))*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    if normalize: psi = psi/psi.norm()
    return psi

# --- cell 41 setup, verbatim ---
Delta = 0.34
N = int(np.floor(0.32/Delta**2))
measIndex = Mmax-4
a = 2*np.sqrt(np.pi); a1 = a/2
U = (1j*(np.pi/2)*aOp1*aOp).expm(); U1 = U.dag()
Hamx = tensor(sigmaz(), a1*xOp)
Hamp = tensor(sigmaz(), -a1*pOp)
Rx = tensor(rx(np.pi/2), qeye(Ncav)); Rx1 = Rx.dag()
DeltaEpsilon = Delta
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} Mmax={Mmax} Delta={Delta} N={N} measIndex={measIndex}", flush=True)

def grid(state, qubit):
    measState = tensor(qubit, state)
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=[], options=Options(nsteps=5000)).states[-1]
    measState = Rx*measState*Rx1
    measState = mesolve(Hamp, measState, [0, np.sqrt(2)*np.cosh(Delta**2)], c_ops=[], options=Options(nsteps=5000)).states[-1]
    measState = Rx1*measState*Rx
    measState = mesolve(Hamx, measState, [0, np.sinh(Delta**2)/np.sqrt(2)], c_ops=[], options=Options(nsteps=10000)).states[-1]
    newState = measState.ptrace(1).unit()
    newqubit = measState.ptrace(0).unit()
    return newState, newqubit

def applyMeasurement(state, correctionDirection, measType='+Z'):
    if 'Z' in measType:
        displacePauli = displace(Ncav, a/4/np.sqrt(2)*np.sinh(DeltaEpsilon**2)+1j*a/4/np.sqrt(2)*np.cosh(DeltaEpsilon**2))
    elif 'X' in measType:
        displacePauli = displace(Ncav, a/4/np.sqrt(2))
    elif 'Y' in measType:
        displacePauli = displace(Ncav, (1+1j)*a/4/np.sqrt(2))
    displacePaulid = displacePauli.dag()
    if '+' in measType: M = 0.5*(displacePauli+displacePaulid)
    elif '-' in measType: M = 0.5*(displacePauli-displacePaulid)
    if correctionDirection == 1: M = displacePauli*M
    else: M = displacePaulid*M
    newState = M*state*M.dag()
    return newState/newState.tr()

state_0 = ket2dm(logical1(0, Delta, N, Ncav))
state_1 = ket2dm(logical1(1, Delta, N, Ncav))
print(f"[{time.time()-t0:.0f}s] binomial L0/L1 built, overlap fidelity={fidelity(state_0,state_1):.4e}", flush=True)

# ============================= (A) grid()-only cooling trace =============================
print("=== (A) grid()-only cooling trace, vacuum start ===", flush=True)
tempState = ket2dm(basis(Ncav,0))
f0, f1 = fidelity(tempState,state_0), fidelity(tempState,state_1)
print(f"round  0: F(L0)={f0:.4f} F(L1)={f1:.4f} max={max(f0,f1):.4f}  [{time.time()-t0:.0f}s]", flush=True)
for i in range(Mmax):
    tempState, tempqubit = grid(tempState, ket2dm(px))
    tempState1, tempqubit = grid(U*tempState*U1, ket2dm(px))
    tempState = U1*tempState1*U
    f0, f1 = fidelity(tempState,state_0), fidelity(tempState,state_1)
    purity = float(np.real((tempqubit*tempqubit).tr()))
    print(f"round {i+1:2d}: F(L0)={f0:.4f} F(L1)={f1:.4f} max={max(f0,f1):.4f}  anc_purity={purity:.4f}  [{time.time()-t0:.0f}s]", flush=True)

# ============================= (B) full computeTrajCooling (+ 1 applyMeasurement pair) =============================
print("=== (B) full computeTrajCooling (grid rounds + applyMeasurement at measIndex) ===", flush=True)
def computeTrajCooling(tempState, Mmax):
    s = [0]
    stateList = [tempState]
    for i in range(Mmax):
        tempState, tempqubit = grid(tempState, ket2dm(px))
        tempState1, tempqubit = grid(U*tempState*U1, ket2dm(px))
        tempState = U1*tempState1*U
        s.append(s[-1] + a*(np.sinh(Delta**2)*np.sqrt(2)+np.cosh(Delta**2)/np.sqrt(2)))
        if i == measIndex:
            tempState = applyMeasurement(tempState, correctionDirection=1)
            tempState = applyMeasurement(tempState, correctionDirection=-1)
            s[-1] = s[-1] + 4*(a/4/np.sqrt(2)*np.sinh(DeltaEpsilon**2)+1j*a/4/np.sqrt(2)*np.cosh(DeltaEpsilon**2))
        if np.mod(i,10) == 0:
            print(f"  SBS ancilla qubit purity after {i+1} steps= {float(np.real((tempqubit*tempqubit).tr())):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        stateList.append(tempState)
    return stateList, s

state0_List, time_sbs = computeTrajCooling(ket2dm(basis(Ncav,0)), Mmax)
state0 = state0_List[-1]     # after Mmax steps
state1 = state0_List[-2]     # after Mmax-1 steps
inf01 = 1-fidelity(state0, state_1)
inf10 = 1-fidelity(state1, state_0)
print(f"[{time.time()-t0:.0f}s] SBS-binomial logical 0 infidelity (state0 vs L1) = {inf01:.4e}  (fidelity={1-inf01:.4f})", flush=True)
print(f"[{time.time()-t0:.0f}s] SBS-binomial logical 1 infidelity (state1 vs L0) = {inf10:.4e}  (fidelity={1-inf10:.4f})", flush=True)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
