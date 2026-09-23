"""Oscillator back-action: reduced-oscillator fidelity vs the ORIGINAL input at m=0, after each
readout (correct within-pulse order). Mean-shift cancelled (readout deterministically displaces).
Compare bare BB1, single-GCR (finite-energy), and the flagged-SUCCESS completed readout."""
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
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]; TARGET=(np.pi/2,0.0)
psi=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
xin=float(np.real(expect(xOp,psi))); pin=float(np.real(expect(pOp,psi)))
def report(tag,st):
    rho=(st.unit()).ptrace(0)
    mx=float(np.real(expect(xOp,rho))); mp=float(np.real(expect(pOp,rho)))
    b=-((mx-xin)+1j*(mp-pin))/np.sqrt(2); D=displace(Ncav,b)
    F=float(np.real((psi.dag()*rho*psi)[0,0])); Fc=float(np.real((psi.dag()*(D*rho*D.dag())*psi)[0,0]))
    print(f"{tag:38s} F_raw={F:.4f}  F_meancancel={Fc:.4f}  (infidelity {1-Fc:.3f})")
# bare BB1: 4 position kicks, no corrections
st=tensor(psi,g)
for th,phi in BB1: st=Opx(th,phi)*st
report("bare BB1", st)
# single-GCR finite-energy readout (target pulse only)
th,phi=TARGET; report("single-GCR (finite-energy)", Opx(th,phi)*(Opy_det(th,phi)*tensor(psi,g)))
# flagged-success completed readout
st=tensor(psi,g)
for th,phi in BB1[:3]: st=Opx(th,phi)*(M_corr(th,phi)*st)
th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*st)
report("flagged-SUCCESS (3 heralds + target)", st)
