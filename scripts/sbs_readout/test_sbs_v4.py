"""Task 0 cont'd: round_p via joint rotation of BOTH oscillator quadrature and qubit frame
(x->p, p->-x, sigma_x->sigma_y, sigma_y->-sigma_x), which preserves the generator's structure."""
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

L0 = logical(0); L0dm = ket2dm(L0)
gq = basis(2,0); gqdm = ket2dm(gq)
def CDgen(vhat, sigma): return (1j*tensor(vhat, sigma)).expm(method='dense')

def round_x():
    small = CDgen(k*alpha*Delta**2*pO, sigmay())
    big = CDgen(k*2*alpha*xO, sigmax())
    return small*big*small

def round_p(sB=1, sS=1):
    small = CDgen(sS*k*alpha*Delta**2*xO, sigmax())
    big   = CDgen(sB*k*2*alpha*pO, sigmay())
    return small*big*small

def run(R, disp, nrounds=10):
    psi0 = (displace(Ncav, disp)*L0).unit()
    rho = tensor(ket2dm(psi0), gqdm)
    fids=[float(np.real(expect(L0dm, rho.ptrace(0))))]
    for n in range(nrounds):
        rho = R*rho*R.dag()
        ro = rho.ptrace(0)
        rho = tensor(ro, gqdm)
        fids.append(float(np.real(expect(L0dm, ro))))
    return fids

print("round_x on pure-x 0.2:", [round(f,4) for f in run(round_x(), 0.2)])
print()
for sB in [1,-1]:
    for sS in [1,-1]:
        fp = run(round_p(sB,sS), 0.2j)
        fx = run(round_p(sB,sS), 0.2)
        print(f"round_p sB={sB:+d} sS={sS:+d}  on pure-p={[round(x,4) for x in fp]}")
        print(f"                          on pure-x={[round(x,4) for x in fx]}")
