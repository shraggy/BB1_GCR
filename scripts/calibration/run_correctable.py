"""Where (if anywhere) do the heralded BB1 corrections beat a single GCR pulse?
Scan the correctable region around the m=0 peak (offset epsilon: m=0 is the peak, m=0.5 the
bit-flip boundary). Compare P(-1) for:
  (A) flag-success = full BB1(GCR)  (3 exact corrections + det GCR target)
  (B) single-GCR finite-energy readout (det GCR target only)
A good (square-wave) readout stays near 0 across the region; BB1 should stay flatter than GCR.
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
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]; TARGET=(np.pi/2,0.0)
def peak(m): al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
def Pm(k): return 1-float(np.real(expect(ket2dm(pyk),k.unit().ptrace(1))))
def full_success(psi):
    st=tensor(psi,g)
    for th,phi in CORR: st=Opx(th,phi)*(M_corr(th,phi)*st)
    th,phi=TARGET; return Opx(th,phi)*(Opy_det(th,phi)*st)
def single_gcr(psi):
    th,phi=TARGET; return Opx(th,phi)*(Opy_det(th,phi)*tensor(psi,g))
print(f"{'offset m':>9} {'full BB1(GCR)':>15} {'single GCR':>12}")
for m in [0.0,0.1,0.2,0.3,0.4,0.45]:
    psi=peak(m)
    print(f"{m:>9} {Pm(full_success(psi)):>15.3e} {Pm(single_gcr(psi)):>12.3e}")
