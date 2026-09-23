"""Can the SAME heralded circuit be used DETERMINISTICALLY (no post-selection) by measuring the
3 ancillas AND the readout qubit and classically decoding the 4-bit outcome into the parity bit?
If yes -> usable end-of-line readout. If it washes out -> the ancilla outcomes don't help.

For each 3-ancilla pattern s and qubit outcome q we get a joint prob P(s,q|m). We build the
BEST FIXED decision rule (report +1 on outcomes more likely under an even peak m=0 than an odd
peak m=1) and report its error at the m=0 peak. Compare to qubit-only (ancillas ignored).
"""
import numpy as np, itertools
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0); pyk=(basis(2,0)+1j*basis(2,1)).unit()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_det(th,phi): return tensor(-1j*(th*ep/sq)*pOp,sigma(phi+np.pi/2)).expm(method='dense')
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def Kbr(th,phi,br): c=th*ep/sq; C=tensor(cos_cp(c),qeye(2)); S=tensor(sin_cp(c),sigma(phi)); return (C+(S if br==0 else -S))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]; TARGET=(np.pi/2,0.0)
def peak(m): al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
Pp=ket2dm(pyk); Pm=qeye(2)-Pp   # qubit projectors: +i -> bit +1, -i -> bit -1
def joint(m):
    """return dict {(s,qbit): prob} for input peak m, over 8 ancilla patterns x 2 qubit outcomes."""
    d={}
    for s in itertools.product([0,1],repeat=3):
        st=tensor(peak(m),g)
        for (th,phi),br in zip(CORR,s): st=Opx(th,phi)*(Kbr(th,phi,br)*st)
        th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*st)   # deterministic target
        ps=float(st.norm()**2); rho_q=(st.unit()).ptrace(1)
        pplus=float(np.real(expect(Pp,rho_q)))               # P(qubit bit=+1 | s)
        d[(s,+1)]=ps*pplus; d[(s,-1)]=ps*(1-pplus)
    return d
d0,d1=joint(0.0),joint(1.0)   # m=0 even (want +1), m=1 odd (want -1)
# best fixed rule: report +1 on outcomes where d0>d1
err_m0=sum(p for o,p in d0.items() if d0[o] < d1[o])   # outcomes ruled -1 but input was m=0
# qubit-only (ignore ancilla pattern): marginalize
q0p=sum(d0[(s,+1)] for s in itertools.product([0,1],repeat=3))
q0m=1-q0p
print(f"deterministic, decode ALL 4 bits (best fixed rule):  P(-1 | m=0) = {err_m0:.3e}")
print(f"deterministic, qubit only (ancillas ignored):        P(-1 | m=0) = {q0m:.3e}")
print(f"(reference: post-selected success branch = 5.7e-4 @ 17%; bare BB1 = 6.1e-3)")
