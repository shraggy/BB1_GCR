"""All-3-fail branch at m=0, CORRECT within-pulse order (correction Kraus first, then position kick).
Report reduced-oscillator fidelity vs input, purity, <x>, and best-displacement-corrected fidelity."""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def K1(th,phi): c=th*ep/sq; return (tensor(cos_cp(c),qeye(2))-tensor(sin_cp(c),sigma(phi)))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]
psi=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
st=tensor(psi,g)
for th,phi in CORR: st=Opx(th,phi)*(K1(th,phi)*st)   # correction first, then kick
rho=(st.unit()).ptrace(0)
F=float(np.real((psi.dag()*rho*psi)[0,0])); pur=float(np.real((rho*rho).tr()))
mx=float(np.real(expect(xOp,rho)))
grid=np.linspace(-1.0,1.0,41); Fbest=F
for re_ in grid:
    for im in grid:
        D=displace(Ncav,complex(re_,im))
        Fc=float(np.real((psi.dag()*(D*rho*D.dag())*psi)[0,0]))
        if Fc>Fbest: Fbest=Fc
print(f"all-fail branch (correct order): F_raw={F:.4f}  purity={pur:.4f}  <x>={mx:+.4f}  <x>_input={float(np.real(expect(xOp,psi))):+.4f}  F_best_disp={Fbest:.4f}")
