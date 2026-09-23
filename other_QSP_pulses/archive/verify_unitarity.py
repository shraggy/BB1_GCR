"""
Verify the user's observation about the BB1(GCR) cell:

  The pre-correction qubit operator passed into exp(-i c * pOp (x) A):
    * vec*vecf  -> NON-unitary gate ("exactly what we want", shows improvement)
    * vec       -> the unitary approximation we could physically build (behaves badly)

Claims checked:
  (C1) vecf^2 = I              (vecf is a rotated Pauli axis)
  (C2) vec*vecf == 1j*sigma(phi)   (anti-Hermitian -> non-unitary generator)
  (C3) sigma(phi) and vecf anticommute => vec = 1j*sigma*vecf is Hermitian (unitary axis)
  (C4) unitarity of the actual two-mode pre-correction operators (||U^dag U - I||)
  (C5) reduced readout sweep: bare BB1 vs BB1(GCR)[vec*vecf] vs BB1(GCR)[vec, unitary]
"""
import numpy as np
from qutip import *

g = basis(2, 0); e = basis(2, 1)
py = (g + 1j * e).unit()
Ncav = 120
aOp = destroy(Ncav); aOp1 = aOp.dag()
i = np.sqrt(2)
xOp = (aOp + aOp1) / i
pOp = (-1j) * (aOp - aOp1) / i

def sigma(phi): return np.cos(phi) * sigmax() + np.sin(phi) * sigmay()
def sigma_xyz(th, phi): return np.cos(th) * sigmaz() + np.sin(th) * sigma(phi)
def rot_xy(th, phi): return (-1j * th / 2 * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay())).expm(method='dense')
def vec_f(v, R): return R * v * R.dag()

Delta = 0.34; r = -np.log(Delta)
theta = np.pi / 2
phi0 = 0; phi1 = np.arccos(-theta / (4 * np.pi)); ep = Delta ** 2
x = np.pi / theta; beta = theta / 2 / (np.sqrt(np.pi) / 2)

def herm_err(A): return (A - A.dag()).norm()
def antiherm_err(A): return (A + A.dag()).norm()

# reconstruct vec / vecf exactly as the cell does, for the first (G4) stage
vecf = vec_f(sigma_xyz(0, 0), rot_xy(0, 0))
vec = 1j * sigma(phi1) * vecf
print("=== qubit-operator algebra (stage G4) ===")
print(f"(C1) ||vecf^2 - I||            = {(vecf*vecf - qeye(2)).norm():.2e}   (expect ~0)")
print(f"(C2) ||vec*vecf - 1j*sigma||   = {(vec*vecf - 1j*sigma(phi1)).norm():.2e}   (expect ~0)")
print(f"     vec*vecf anti-Hermitian?  ||A+A^dag|| = {antiherm_err(vec*vecf):.2e}  Hermitian? ||A-A^dag||={herm_err(vec*vecf):.2e}")
print(f"(C3) {{sigma(phi1), vecf}} anticommute? ||sig*vecf+vecf*sig|| = {(sigma(phi1)*vecf+vecf*sigma(phi1)).norm():.2e}")
print(f"     vec Hermitian? ||vec-vec^dag|| = {herm_err(vec):.2e}  (Hermitian -> unitary CD axis)")

# ---- build the two-mode pre-correction operators and test unitarity ----
G4_nonU = tensor(-1j * x * beta * (ep * pOp), vec * vecf).expm(method='dense')   # vec*vecf
G4_U    = tensor(-1j * x * beta * (ep * pOp), vec).expm(method='dense')          # vec  (physical)
I2 = tensor(qeye(Ncav), qeye(2))
print("\n=== unitarity of the pre-correction gate exp(-i c pOp (x) A) ===")
print(f"(C4) A=vec*vecf : ||U^dag U - I|| = {(G4_nonU.dag()*G4_nonU - I2).norm():.3e}   <- NON-unitary")
print(f"     A=vec      : ||U^dag U - I|| = {(G4_U.dag()*G4_U - I2).norm():.3e}   <- unitary")

# ---- build full 4-stage operators for both modes (alpha-independent) ----
def build_stages(mode):
    """mode: 'vecvecf' (notebook), 'vec' (physical unitary)."""
    A = lambda v, vf: (v * vf) if mode == 'vecvecf' else v
    out = []
    vecf = vec_f(sigma_xyz(0, 0), rot_xy(0, 0)); vec = 1j * sigma(phi1) * vecf
    out.append(tensor(-1j*x*beta*xOp, sigma(phi1)).expm(method='dense') * tensor(-1j*x*beta*(ep*pOp), A(vec, vecf)).expm(method='dense'))
    vecf = vec_f(vecf, rot_xy(2*np.pi, phi1)); vec = 1j*sigma(3*phi1)*vecf
    out.append(tensor(-2j*np.sqrt(np.pi)*xOp, sigma(3*phi1)).expm(method='dense') * tensor(-2j*np.sqrt(np.pi)*(ep*pOp), A(vec, vecf)).expm(method='dense'))
    vecf = vec_f(vecf, rot_xy(4*np.pi, 3*phi1)); vec = 1j*sigma(phi1)*vecf
    out.append(tensor(-1j*np.sqrt(np.pi)*xOp, sigma(phi1)).expm(method='dense') * tensor(-1j*np.sqrt(np.pi)*(ep*pOp), A(vec, vecf)).expm(method='dense'))
    vecf = vec_f(vecf, rot_xy(2*np.pi, phi1)); vec = 1j*sigma(phi0)*vecf
    out.append(tensor(-1j*beta*xOp, sigma(phi0)).expm(method='dense') * tensor(-1j*beta*(ep*pOp), A(vec, vecf)).expm(method='dense'))
    return out

ST_vecvecf = build_stages('vecvecf')
ST_vec = build_stages('vec')
B4 = tensor(1j*np.sqrt(np.pi)*xOp, sigma(phi1)).expm(method='dense').dag()
B3 = tensor(2j*np.sqrt(np.pi)*xOp, sigma(3*phi1)).expm(method='dense').dag()
B2 = tensor(1j*np.sqrt(np.pi)*xOp, sigma(phi1)).expm(method='dense').dag()
B1 = tensor(1j*beta*xOp, sigma(phi0)).expm(method='dense').dag()
BB1_ST = [B4, B3, B2, B1]

def run(stages, init):
    s = tensor(init, g)
    for U in stages:
        s = (U * s).unit()
    return s

print("\n=== reduced readout sweep: P(+1) response ===")
N = 31
alph = np.linspace(-2*np.sqrt(np.pi), 2*np.sqrt(np.pi), N)
a = -np.sqrt(np.pi)/2
rows = []
for alpha1 in alph:
    alpha = alpha1 + a
    init = (displace(Ncav, alpha/np.sqrt(2)) * (squeeze(Ncav, r) * basis(Ncav, 0)).unit()).unit()
    p_bb1 = np.real(expect(ket2dm(py), run(BB1_ST, init).ptrace(1)))
    p_nonU = np.real(expect(ket2dm(py), run(ST_vecvecf, init).ptrace(1)))
    p_U = np.real(expect(ket2dm(py), run(ST_vec, init).ptrace(1)))
    rows.append((alpha1/np.sqrt(np.pi), p_bb1, p_nonU, p_U))

rows = np.array(rows)
print("   m=<x>/sqrt(pi)   BB1     BB1(GCR)vec*vecf[nonU]   BB1(GCR)vec[unitary]")
for k in range(0, N, 2):
    print(f"   {rows[k,0]:+6.2f}        {rows[k,1]:.3f}        {rows[k,2]:.3f}                  {rows[k,3]:.3f}")

# how "square" / clean is each response: variation of P within the central plateau bins
def plateau_quality(p, m):
    # crude: std of P about its rounded {0,1} target across all points
    tgt = np.round(p)
    return np.mean(np.abs(p - tgt))
print("\nmean |P - nearest(0/1)|  (smaller = cleaner square-wave readout):")
print(f"   bare BB1                 : {plateau_quality(rows[:,1], rows[:,0]):.3f}")
print(f"   BB1(GCR) vec*vecf [nonU] : {plateau_quality(rows[:,2], rows[:,0]):.3f}")
print(f"   BB1(GCR) vec      [unit] : {plateau_quality(rows[:,3], rows[:,0]):.3f}")
np.savez("Paper_Data/unitarity_sweep.npz", rows=rows)
print("saved Paper_Data/unitarity_sweep.npz")
