"""Per-BB1-angle comparison: how well does the first-order gadget (and hence any unitary) approximate
the exact non-unitary correction M=e^{c p sigma_phi} at each BB1 angle (pi, 2pi, pi/2)?
Fidelity of the gadget success state K0|psi> vs the exact M|psi> on the m=0 peak."""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
psi=tensor((displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit(),g)
print(f"{'BB1 angle':>10} {'axis phi':>9} {'c=eta*Delta^2/2|a|':>18} {'fidelity(gadget,exact M)':>26}")
for name,eta,phi in [('pi',np.pi,phi1),('2pi',2*np.pi,3*phi1),('pi/2 (target)',np.pi/2,0.0)]:
    c=eta*ep/sq
    K0=(tensor(cos_cp(c),qeye(2))+tensor(sin_cp(c),sigma(phi)))/np.sqrt(2)
    M =tensor(c*pOp,sigma(phi)).expm(method='dense')
    exact=(M*psi).unit(); gad=(K0*psi).unit()
    F=float(np.abs(exact.overlap(gad))**2)
    print(f"{name:>10} {phi:>9.2f} {c:>18.3f} {F:>26.4f}")
