"""Does the MODULAR-POSITION info survive a failed herald, so we can re-read?
Full-state fidelity is only 0.69, but a modular readout only needs the peak identity
(which peak -> which bit). Test directly:
 - build the all-3-FAIL reduced oscillator state at each peak m
 - report its <x> vs the input's <x> (is the mean position preserved per peak?)
 - RE-READ it: apply a fresh bare-BB1 readout to rho_osc (x) |g> and compare the
   response P(-1) to reading the pristine input. Correct pattern: P(-1) small at even
   floor(m) (bit +1), ~1 at odd floor(m) (bit -1). If the failed state still yields the
   correct bits, the info survived and a retry works (just noisier).
"""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
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
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def input_osc(m):
    al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
def bare_readout():
    U=None
    for th,phi in BB1:
        op=Opx(th,phi); U=op if U is None else op*U
    return U
Ubare=bare_readout()
def fail_rho_osc(m):
    st=tensor(input_osc(m),g)
    for th,phi in CORR: st=Kraus(th,phi,1)*(Opx(th,phi)*st)
    return (st.unit()).ptrace(0)
def read_pure(psi):   # P(-1) reading a pure oscillator state
    out=(Ubare*tensor(psi,g)).unit()
    return 1-float(np.real(expect(ket2dm(py),out.ptrace(1))))
def read_rho(rho):    # P(-1) reading a mixed oscillator state
    R=tensor(rho,ket2dm(g)); R=Ubare*R*Ubare.dag()
    return 1-float(np.real(expect(ket2dm(py),R.ptrace(1))))

print(f"{'m':>4} {'<x>_in':>9} {'<x>_fail':>9} {'P(-1)_input':>12} {'P(-1)_reread_failed':>20}  bit")
for m in [0.0,1.0,2.0,3.0]:
    psi=input_osc(m); rho=fail_rho_osc(m)
    xin=float(np.real(expect(xOp,psi))); xf=float(np.real(expect(xOp,rho)))
    p_in=read_pure(psi); p_re=read_rho(rho)
    correct='+1' if int(np.floor(m))%2==0 else '-1'
    print(f"{m:>4} {xin:>9.4f} {xf:>9.4f} {p_in:>12.3e} {p_re:>20.3e}   want {correct}")
