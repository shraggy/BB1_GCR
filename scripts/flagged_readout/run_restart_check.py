"""Is the flagged readout recoverable on a FAILED herald? i.e. if the three heralded
corrections do NOT all succeed, can we restart the readout with the (corrected) same
oscillator state?

Model: each heralded correction has success Kraus K0=(cos(cp)+sin(cp)sigma_phi)/sqrt2
and failure Kraus K1=(cos(cp)-sin(cp)sigma_phi)/sqrt2, with A=c p sigma_phi,
c=theta*Delta^2/sqrt(pi). (A^2 = c^2 p^2 (x) I, so cos A = cos(cp)(x)I,
sin A = sin(cp)(x)sigma_phi.) The x-controlled rotation Opx acts regardless of herald.

We compute, for the all-success and all-fail branches of the 3 corrections:
 - branch probability
 - reduced oscillator state fidelity vs the ORIGINAL input (raw)
 - best fidelity after an optimal corrective displacement D(beta) (momentum/position boost)
 - purity of the reduced oscillator state (1 => disentangled & pure => cleanly reusable)
If F_corrected ~ 1 and purity ~ 1 on the fail branch, restart-with-same-state works.
"""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay()
g=basis(2,0)
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def cos_cp(c):
    U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c):
    U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def Kraus(th,phi,branch):
    c=th*ep/sq
    C=tensor(cos_cp(c),qeye(2)); S=tensor(sin_cp(c),sigma(phi))
    return (C+(S if branch==0 else -S))/np.sqrt(2)

CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]   # the three correction pulses only

def input_osc(m):
    al=m*sq
    return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()

# corrective-displacement grid
betas=[complex(re,im) for re in np.linspace(-0.7,0.7,15) for im in np.linspace(-0.7,0.7,15)]
Dops=[displace(Ncav,b) for b in betas]

for m in [0.0,1.0]:
    psi_in=input_osc(m); rho_in=ket2dm(psi_in)
    psi0=tensor(psi_in,g)
    for branch,name in [(0,'all-SUCCESS'),(1,'all-FAIL')]:
        st=psi0
        for th,phi in CORR:
            st=Kraus(th,phi,branch)*(Opx(th,phi)*st)
        p=float(st.norm()**2); st=st.unit()
        rho_osc=st.ptrace(0)
        F_raw=float(np.real((psi_in.dag()*rho_osc*psi_in)[0,0]))
        pur=float(np.real((rho_osc*rho_osc).tr()))
        F_best=0.0; bb=0
        for D,b in zip(Dops,betas):
            Fc=float(np.real((psi_in.dag()*(D*rho_osc*D.dag())*psi_in)[0,0]))
            if Fc>F_best: F_best=Fc; bb=b
        print(f"m={m:>3} {name:12s} P={p:6.3f}  F_raw={F_raw:7.4f}  "
              f"F_corr={F_best:7.4f} (beta={bb:.2f})  purity={pur:7.4f}")
