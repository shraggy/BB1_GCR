import numpy as np
from qutip import *

Ncav = 100
aC = destroy(Ncav); xO = (aC + aC.dag())/np.sqrt(2); pO = (-1j)*(aC - aC.dag())/np.sqrt(2)
Delta = 0.34; r = -np.log(Delta)
alpha = np.sqrt(np.pi/2)/2
k = np.sqrt(2)

def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()

L0 = logical(0); L1 = logical(1)
L0dm = ket2dm(L0); L1dm = ket2dm(L1)
gq = basis(2,0); gqdm = ket2dm(gq)
def CDgen(vhat, sigma): return (1j*tensor(vhat, sigma)).expm(method='dense')

small = CDgen(k*alpha*Delta**2*xO, sigmay())
big   = CDgen(k*2*alpha*pO, sigmax())
Rp = small*big*small

rho = tensor(L0dm, gqdm)
print("BEFORE: F(L0)=",np.real(expect(L0dm, rho.ptrace(0))), " F(L1)=",np.real(expect(L1dm, rho.ptrace(0))))
rho2 = Rp*rho*Rp.dag()
ro = rho2.ptrace(0)
print("qubit reduced state after Rp:", rho2.ptrace(1))
print("AFTER Rp on ideal L0 (before tracing qubit): F_osc(L0)=", np.real(expect(L0dm, ro)), " F_osc(L1)=", np.real(expect(L1dm, ro)), " purity=", np.real((ro*ro).tr()))

# now check the g/e branch-conditioned oscillator states
Pg = tensor(qeye(Ncav), ket2dm(basis(2,0)))
Pe = tensor(qeye(Ncav), ket2dm(basis(2,1)))
rho_g = Pg*rho2*Pg; pg = rho_g.tr(); rho_g = rho_g/pg if pg>1e-12 else rho_g
rho_e = Pe*rho2*Pe; pe = rho_e.tr(); rho_e = rho_e/pe if pe>1e-12 else rho_e
print(f"p(g)={pg:.4f} p(e)={pe:.4f}")
print("F(L0|g)=", np.real(expect(L0dm, rho_g.ptrace(0))), "F(L1|g)=", np.real(expect(L1dm, rho_g.ptrace(0))))
print("F(L0|e)=", np.real(expect(L0dm, rho_e.ptrace(0))), "F(L1|e)=", np.real(expect(L1dm, rho_e.ptrace(0))))
