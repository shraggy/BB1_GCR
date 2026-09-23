"""Task 0 refinement: isolate x-only and p-only errors to determine correct round_p."""
import numpy as np
from qutip import *

Ncav = 100
aC = destroy(Ncav); xO = (aC + aC.dag())/np.sqrt(2); pO = (-1j)*(aC - aC.dag())/np.sqrt(2)
Delta = 0.34; r = -np.log(Delta)
alpha = np.sqrt(np.pi/2)/2
lam = -alpha*Delta**2

def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()

L0 = logical(0); L0dm = ket2dm(L0)
gq = basis(2,0); gqdm = ket2dm(gq)

def CDop(beta, sigma, k, sign):
    vhat = sign*k*(np.imag(beta)*xO - np.real(beta)*pO)
    return (1j*tensor(vhat, sigma)).expm(method='dense')

k = np.sqrt(2)

def round_x(sign=1):
    small = CDop(lam, sigmay(), k, sign)
    big = CDop(1j*2*alpha, sigmax(), k, sign)
    return small*big*small

# Candidate round_p variants
def round_p_literal(sign=1):
    # user's literal dual formula: CD(i*lambda,sigma_x)*CD(2*alpha,sigma_y)*CD(i*lambda,sigma_x)
    small = CDop(1j*lam, sigmax(), k, sign)
    big = CDop(2*alpha, sigmay(), k, sign)
    return small*big*small

def round_p_rot(sign=1):
    # rotation-consistent: x->p, p->-x, same Pauli labels (sigma_x for big, sigma_y for small)
    small = CDop(-1j*lam, sigmay(), k, sign)   # from -i*k*alpha*Delta^2*xhat*sigma_y => beta s.t. Im(beta)=-lam... let's just build directly
    big = CDop(2*alpha, sigmax(), k, sign)     # want exp(i*k*2*alpha*phat*sigma_x): Im(beta)xhat-Re(beta)phat = phat*2alpha => Re(beta)=-2alpha
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

print("=== round_x on pure x-error (disp=0.2) ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_x(sign), 0.2)]}")

print("=== round_x on pure p-error (disp=0.2j) [should NOT correct] ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_x(sign), 0.2j)]}")

print("=== round_p_literal on pure p-error (disp=0.2j) ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_literal(sign), 0.2j)]}")

print("=== round_p_literal on pure x-error (disp=0.2) [should NOT correct] ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_literal(sign), 0.2)]}")

print("=== round_p_rot on pure p-error (disp=0.2j) ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_rot(sign), 0.2j)]}")

print("=== round_p_rot on pure x-error (disp=0.2) [should NOT correct] ===")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_rot(sign), 0.2)]}")

print("\n=== round_p_v2 (properly derived: small on sigma_y w/ beta=i*lam, big on sigma_x w/ beta=-2*alpha) ===")
def round_p_v2(sign=1):
    small = CDop(1j*lam, sigmay(), k, sign)
    big = CDop(-2*alpha, sigmax(), k, sign)
    return small*big*small

print("--- pure p-error (disp=0.2j) ---")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_v2(sign), 0.2j)]}")
print("--- pure x-error (disp=0.2) [should NOT correct / stay flat] ---")
for sign in [1,-1]:
    print(f"sign={sign}: {[round(f,4) for f in run(round_p_v2(sign), 0.2)]}")

print("\n=== combined: alternate round_x, round_p_v2 on MIXED error 0.15+0.12j ===")
def run_alt(Rx, Rp, disp, nrounds=10):
    psi0 = (displace(Ncav, disp)*L0).unit()
    rho = tensor(ket2dm(psi0), gqdm)
    fids=[float(np.real(expect(L0dm, rho.ptrace(0))))]
    impurs=[1-float(np.real((rho.ptrace(0)*rho.ptrace(0)).tr()))]
    for n in range(nrounds):
        R = Rx if n%2==0 else Rp
        rho = R*rho*R.dag()
        ro = rho.ptrace(0)
        rho = tensor(ro, gqdm)
        fids.append(float(np.real(expect(L0dm, ro))))
        impurs.append(1-float(np.real((ro*ro).tr())))
    return fids, impurs

for sign in [1,-1]:
    f,im = run_alt(round_x(sign), round_p_v2(sign), 0.15+0.12j)
    print(f"sign={sign}: F={[round(x,4) for x in f]}")
    print(f"          impur={[f'{x:.2e}' for x in im]}")
