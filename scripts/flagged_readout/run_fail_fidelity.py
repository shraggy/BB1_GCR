"""Fidelity of the oscillator (reduced) state vs the ORIGINAL input, after the 3 correction
heralds, grouped by how many of the three heralds FAILED. At the m=0 peak. F = <psi_in| rho_osc |psi_in>.
Also branch probability and oscillator purity (1 = disentangled from the readout qubit)."""
import numpy as np, itertools
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def Kbr(th,phi,br): c=th*ep/sq; C=tensor(cos_cp(c),qeye(2)); S=tensor(sin_cp(c),sigma(phi)); return (C+(S if br==0 else -S))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]
psi_in=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()  # m=0 peak
print(f"{'ancilla pattern':>16} {'#fail':>6} {'prob':>8} {'F(osc,input)':>13} {'purity':>8}")
for s in itertools.product([0,1],repeat=3):   # 0=herald success, 1=herald fail
    st=tensor(psi_in,g)
    for (th,phi),br in zip(CORR,s): st=Opx(th,phi)*(Kbr(th,phi,br)*st)
    p=float(st.norm()**2); rho=(st.unit()).ptrace(0)
    F=float(np.real((psi_in.dag()*rho*psi_in)[0,0])); pur=float(np.real((rho*rho).tr()))
    print(f"{str(s):>16} {sum(s):>6} {p:>8.3f} {F:>13.4f} {pur:>8.4f}")
