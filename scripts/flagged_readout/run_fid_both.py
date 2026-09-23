"""Reduced-oscillator fidelity vs the ORIGINAL input at m=0, correct within-pulse order.
(A) all-3-fail branch (3 anti-correction Krauss)
(B) completed POST-SELECTED readout: 3 successful M-corrections + deterministic GCR target (4th pulse)
Report F_raw, purity, <x>,<p>, and F after cancelling the mean shift (single analytic displacement),
since a readout deterministically displaces the state by ~|alpha| in the conjugate quadrature."""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def M_corr(th,phi): return tensor((th*ep/sq)*pOp,sigma(phi)).expm(method='dense')
def Opy_det(th,phi): return tensor(-1j*(th*ep/sq)*pOp,sigma(phi+np.pi/2)).expm(method='dense')
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def K1(th,phi): c=th*ep/sq; return (tensor(cos_cp(c),qeye(2))-tensor(sin_cp(c),sigma(phi)))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]; TARGET=(np.pi/2,0.0)
psi=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
xin=float(np.real(expect(xOp,psi))); pin=float(np.real(expect(pOp,psi)))
def report(tag,rho):
    F=float(np.real((psi.dag()*rho*psi)[0,0])); pur=float(np.real((rho*rho).tr()))
    mx=float(np.real(expect(xOp,rho))); mp=float(np.real(expect(pOp,rho)))
    b=-((mx-xin)+1j*(mp-pin))/np.sqrt(2); D=displace(Ncav,b)
    Fc=float(np.real((psi.dag()*(D*rho*D.dag())*psi)[0,0]))
    print(f"{tag:42s} F_raw={F:.4f}  purity={pur:.4f}  <x>={mx:+.3f} <p>={mp:+.3f}  F_meancancel={Fc:.4f}")
# (A) all-fail
st=tensor(psi,g)
for th,phi in CORR: st=Opx(th,phi)*(K1(th,phi)*st)
report("(A) all-3-fail (3 corrections only)", (st.unit()).ptrace(0))
# (B) completed post-selected readout
st=tensor(psi,g)
for th,phi in CORR: st=Opx(th,phi)*(M_corr(th,phi)*st)
th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*st)
report("(B) success + target (completed readout)", (st.unit()).ptrace(0))
