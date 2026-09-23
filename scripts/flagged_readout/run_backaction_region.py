"""Oscillator back-action (reduced-osc fidelity vs input, mean-cancelled) across the correctable
region m=0..0.4, for bare BB1 vs single-GCR vs flagged-SUCCESS completed readout."""
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
def peak(m): al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
def Fback(psi,st):
    xin=float(np.real(expect(xOp,psi))); pin=float(np.real(expect(pOp,psi)))
    rho=(st.unit()).ptrace(0); mx=float(np.real(expect(xOp,rho))); mp=float(np.real(expect(pOp,rho)))
    D=displace(Ncav,-((mx-xin)+1j*(mp-pin))/np.sqrt(2))
    return float(np.real((psi.dag()*(D*rho*D.dag())*psi)[0,0]))
def bare(psi):
    st=tensor(psi,g)
    for th,phi in BB1: st=Opx(th,phi)*st
    return st
def gcr(psi):
    th,phi=TARGET; return Opx(th,phi)*(Opy_det(th,phi)*tensor(psi,g))
def flag(psi):
    st=tensor(psi,g)
    for th,phi in BB1[:3]: st=Opx(th,phi)*(M_corr(th,phi)*st)
    th,phi=TARGET; return Opx(th,phi)*(Opy_det(th,phi)*st)
print(f"{'m':>5} {'bare BB1':>10} {'single-GCR':>11} {'flag-success':>13}")
for m in [0.0,0.1,0.2,0.3,0.4]:
    psi=peak(m)
    print(f"{m:>5} {Fback(psi,bare(psi)):>10.4f} {Fback(psi,gcr(psi)):>11.4f} {Fback(psi,flag(psi)):>13.4f}")
