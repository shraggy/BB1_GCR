"""Task 0 continued: sweep relative signs of small vs big in the p-stabilizing round to find
the true stabilizer (logical IDENTITY, not logical Pauli flip)."""
import numpy as np
from qutip import *

Ncav = 100
aC = destroy(Ncav); xO = (aC + aC.dag())/np.sqrt(2); pO = (-1j)*(aC - aC.dag())/np.sqrt(2)
Delta = 0.34; r = -np.log(Delta)
alpha = np.sqrt(np.pi/2)/2
lam = -alpha*Delta**2
k = np.sqrt(2)

def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()

L0 = logical(0); L0dm = ket2dm(L0)
gq = basis(2,0); gqdm = ket2dm(gq)

def CDgen(vhat, sigma):
    return (1j*tensor(vhat, sigma)).expm(method='dense')

def round_x():
    small = CDgen(alpha*Delta**2*pO, sigmay())   # vhat = +alpha*Delta^2 * p  (validated k=sqrt2 folded into alpha*Delta^2 via k factor)
    # NOTE: fold k into the coefficient directly for clarity: coefficient = k*alpha*Delta^2
    small = CDgen(k*alpha*Delta**2*pO, sigmay())
    big = CDgen(k*2*alpha*xO, sigmax())
    return small*big*small

def round_p(s_small, s_big):
    small = CDgen(s_small*k*alpha*Delta**2*xO, sigmay())
    big = CDgen(s_big*k*2*alpha*pO, sigmax())
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

print("round_x sanity on pure-x error 0.2:", [round(f,4) for f in run(round_x(), 0.2)])
print()
for s_small in [1,-1]:
    for s_big in [1,-1]:
        f = run(round_p(s_small,s_big), 0.2j)
        print(f"round_p s_small={s_small:+d} s_big={s_big:+d}  pure-p-error F={[round(x,4) for x in f]}")
