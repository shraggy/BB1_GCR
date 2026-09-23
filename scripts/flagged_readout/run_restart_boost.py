"""Try to recover the failed-herald oscillator state with a DISPLACEMENT / MOMENTUM BOOST.
Recipe (user's): calibrate the best displacement beta* at <x>=0, then apply that SAME beta*
to every peak and check fidelity. Also report:
 - the mean shift <x>,<p> of the failed state (what a displacement could naively undo)
 - lambda_max(rho_osc): the hard ceiling on <psi|U rho U^dag|psi> for ANY oscillator unitary U
   (if the state is entangled with the qubit, purity<1 caps the achievable fidelity)
 - a qubit-CONDITIONAL displacement (best case: disentangle then boost) for comparison.
"""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz(); g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def cos_cp(c):
    U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c):
    U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def Kraus(th,phi,branch):
    c=th*ep/sq; C=tensor(cos_cp(c),qeye(2)); S=tensor(sin_cp(c),sigma(phi))
    return (C+(S if branch==0 else -S))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]
def input_osc(m):
    al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
def fail_state(m):
    st=tensor(input_osc(m),g)
    for th,phi in CORR: st=Kraus(th,phi,1)*(Opx(th,phi)*st)
    return st.unit()

# --- calibrate best displacement at <x>=0 (fine grid) ---
st0=fail_state(0.0); rho0=st0.ptrace(0); psi0=input_osc(0.0)
mx=float(np.real(expect(xOp,rho0))); mp=float(np.real(expect(pOp,rho0)))
evals=rho0.eigenenergies(); lam_max=float(evals[-1])
print(f"failed state @x=0:  <x>={mx:+.4f}  <p>={mp:+.4f}  lambda_max(rho)={lam_max:.4f}  (fidelity ceiling for ANY osc unitary)")
grid=np.linspace(-1.2,1.2,49)
best=(-1,0j)
for re in grid:
    for im in grid:
        b=complex(re,im); D=displace(Ncav,b)
        F=float(np.real((psi0.dag()*(D*rho0*D.dag())*psi0)[0,0]))
        if F>best[0]: best=(F,b)
print(f"best uniform displacement @x=0:  beta*={best[1]:+.3f}  F={best[0]:.4f}")
# displacement that cancels the mean shift, for reference: beta=-(mx+i mp)/sqrt2
b_mean=-(mx+1j*mp)/np.sqrt(2); Dm=displace(Ncav,b_mean)
Fm=float(np.real((psi0.dag()*(Dm*rho0*Dm.dag())*psi0)[0,0]))
print(f"mean-cancelling displacement:     beta ={b_mean:+.3f}  F={Fm:.4f}")

# --- apply the SAME beta* to every peak ---
bstar=best[1]; D=displace(Ncav,bstar)
print("\napply calibrated beta* uniformly across peaks:")
for m in [0.0,1.0,2.0]:
    st=fail_state(m); rho=st.ptrace(0); psi=input_osc(m)
    Fraw=float(np.real((psi.dag()*rho*psi)[0,0]))
    Fc  =float(np.real((psi.dag()*(D*rho*D.dag())*psi)[0,0]))
    print(f"  m={m:>3}  F_raw={Fraw:.4f}  F_afterBoost={Fc:.4f}")

# --- best-case: qubit-CONDITIONAL displacement, via Schmidt decomposition ---
# rho_osc eigenvectors |phi_i> (weights p_i) are the Schmidt oscillator states, each paired
# with an orthonormal qubit state. A qubit-conditional displacement applies D_i to |phi_i>;
# best reduced fidelity = sum_i p_i * max_D |<psi|D|phi_i>|^2. This is the BEST any conditional
# oscillator displacement can do (still no re-entangling with the lost ancilla info).
print("\nbest-case qubit-conditional displacement (Schmidt):")
Dgrid=[displace(Ncav,complex(re,im)) for re in grid for im in grid]
ev,evec=rho0.eigenstates()
F_cond=0.0
for p,phi in list(zip(ev,evec))[-2:]:      # two dominant Schmidt terms (qubit is 2-dim)
    best_i=max(float(np.abs((psi0.dag()*(D*phi))[0,0])**2) for D in Dgrid)
    F_cond+=float(p)*best_i
    print(f"  Schmidt weight p={float(p):.4f}  best |<psi|D|phi>|^2={best_i:.4f}")
print(f"  => best conditional fidelity = {F_cond:.4f}")
