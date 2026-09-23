"""Can you restart the readout on the SAME (disturbed) oscillator with a FRESH qubit?
(You cannot 'start from the original' -- unknown data, no copy to re-prepare.)
Model k successive FAILED attempts (each: 3 correction-gadgets fail -> apply K1 to osc+qubit,
then trace/reset the qubit), then ask on the disturbed oscillator rho_k:
  * <x> at the m=0 peak  (is the modular/peak info still there?)
  * a fresh flag-SUCCESS read  P(-1)@0  (if this restart finally succeeds, is the bit still sharp?)
  * a fresh single-GCR read P(-1)@0     (degraded fallback quality)
"""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0); pyk=(basis(2,0)+1j*basis(2,1)).unit()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def M_corr(th,phi): return tensor((th*ep/sq)*pOp,sigma(phi)).expm(method='dense')
def Opy_det(th,phi): return tensor(-1j*(th*ep/sq)*pOp,sigma(phi+np.pi/2)).expm(method='dense')
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def K1(th,phi): c=th*ep/sq; return (tensor(cos_cp(c),qeye(2))-tensor(sin_cp(c),sigma(phi)))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]; TARGET=(np.pi/2,0.0)
def peak(m): al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
def Pm_rho(R): return 1-float(np.real(expect(ket2dm(pyk),R.ptrace(1))))
def fresh(rho_osc): return tensor(rho_osc, ket2dm(g))
def one_failed_round(rho_osc):     # 3 corrections all fail, qubit reset afterwards
    R=fresh(rho_osc)
    for th,phi in CORR:
        R=Opx(th,phi)*(K1(th,phi)*R*K1(th,phi).dag())*Opx(th,phi).dag()
    R=R/R.tr(); return R.ptrace(0)
def read_success(rho_osc):         # fresh flag-success read on disturbed oscillator
    R=fresh(rho_osc)
    for th,phi in CORR: R=Opx(th,phi)*(M_corr(th,phi)*R*M_corr(th,phi).dag())*Opx(th,phi).dag()
    th,phi=TARGET; R=Opx(th,phi)*(Opy_det(th,phi)*R*Opy_det(th,phi).dag())*Opx(th,phi).dag()
    R=R/R.tr(); return Pm_rho(R)
def read_gcr(rho_osc):             # fresh single-GCR read on disturbed oscillator
    R=fresh(rho_osc); th,phi=TARGET
    R=Opx(th,phi)*(Opy_det(th,phi)*R*Opy_det(th,phi).dag())*Opx(th,phi).dag(); R=R/R.tr(); return Pm_rho(R)

rho=ket2dm(peak(0.0))
print(f"{'k failed':>9} {'<x>':>9} {'success-read P(-1)@0':>21} {'single-GCR read P(-1)@0':>24}")
for k in range(0,4):
    x=float(np.real((xOp*rho).tr()))
    print(f"{k:>9} {x:>9.4f} {read_success(rho):>21.3e} {read_gcr(rho):>24.3e}")
    rho=one_failed_round(rho)
