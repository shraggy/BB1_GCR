"""What does each flag outcome actually read out?
Compare, at <x>=0:
  (1) flag SUCCESS  = all 3 correction-heralds give A=0  -> full ideal BB1(GCR)   (expect 5.7e-4)
  (2) flag FAIL     = all 3 heralds give A=1 (complementary Kraus) + det. GCR target
  (3) single-GCR "finite-energy" readout = just the deterministic target pulse (Opx*Opy_det)
  (4) flag-IGNORED  = run the 3 gadgets but TRACE the ancillas (no post-selection) + det. target
  (5) bare BB1 (reference)                                                          (=6.1e-3)
This tells us whether a flagged-failed shot is a finite-energy-GCR-quality readout, bare-level,
or worse -- i.e. what 'the flag' operationally selects between.
"""
import numpy as np
from qutip import *
Ncav=110
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=-1j*(aOp-aOp.dag())/np.sqrt(2)
SX,SY=sigmax(),sigmay(); g=basis(2,0); pyk=(basis(2,0)+1j*basis(2,1)).unit()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8); a=-sq/2
def Opx(th,phi):     return tensor(-1j*(th/sq)*xOp, sigma(phi)).expm(method='dense')
def M_corr(th,phi):  return tensor((th*ep/sq)*pOp, sigma(phi)).expm(method='dense')       # ideal (success)
def Opy_det(th,phi): return tensor(-1j*(th*ep/sq)*pOp, sigma(phi+np.pi/2)).expm(method='dense')  # det. GCR
def cos_cp(c): U=((1j*c)*pOp).expm(method='dense'); return 0.5*(U+U.dag())
def sin_cp(c): U=((1j*c)*pOp).expm(method='dense'); return (U-U.dag())/(2j)
def Kbranch(th,phi,br): c=th*ep/sq; C=tensor(cos_cp(c),qeye(2)); S=tensor(sin_cp(c),sigma(phi)); return (C+(S if br==0 else -S))/np.sqrt(2)
CORR=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1)]; TARGET=(np.pi/2,0.0)
def peak(m): al=m*sq; return (displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
psi0=peak(0.0)
def Pminus_from_ket(k): return 1-float(np.real(expect(ket2dm(pyk), k.unit().ptrace(1))))
def Pminus_from_rho(R): return 1-float(np.real(expect(ket2dm(pyk), R.ptrace(1))))

# within each GCR unit the operator is Opx * (correction): correction acts FIRST, then kick.
# (1) flag success: exact M on the 3 corrections + det target
st=tensor(psi0,g)
for th,phi in CORR: st=Opx(th,phi)*(M_corr(th,phi)*st)
th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*st)
print(f"(1) flag SUCCESS (full ideal BB1(GCR))     P(-1)@0 = {Pminus_from_ket(st):.3e}")

# (2) flag fail: complementary Kraus (br=1) on the 3 corrections + det target
st=tensor(psi0,g)
for th,phi in CORR: st=Opx(th,phi)*(Kbranch(th,phi,1)*st)
th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*st)
print(f"(2) flag FAIL (anti-corrections + det GCR) P(-1)@0 = {Pminus_from_ket(st):.3e}")

# (3) single-GCR finite-energy readout = just the deterministic target pulse
th,phi=TARGET; st=Opx(th,phi)*(Opy_det(th,phi)*tensor(psi0,g))
print(f"(3) single-GCR finite-energy readout       P(-1)@0 = {Pminus_from_ket(st):.3e}")

# (4) flag IGNORED: run the 3 gadgets as a channel (sum over both branches), trace ancillas, + det target
R=ket2dm(tensor(psi0,g))
for th,phi in CORR:
    K0=Kbranch(th,phi,0); K1=Kbranch(th,phi,1)
    R=K0*R*K0.dag()+K1*R*K1.dag()          # correction channel first (CPTP, no post-selection)
    U=Opx(th,phi); R=U*R*U.dag()           # then position kick
th,phi=TARGET; R=Opy_det(th,phi)*R*Opy_det(th,phi).dag(); U=Opx(th,phi); R=U*R*U.dag()
R=R/R.tr()
print(f"(4) flag IGNORED (deterministic, no PS)    P(-1)@0 = {Pminus_from_rho(R):.3e}")

print(f"(5) bare BB1 (reference)                   P(-1)@0 = 6.1e-3")
