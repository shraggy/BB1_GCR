"""Track logical Pauli back-action: round_x back-action is logical Z (invisible on |0>), round_p's
is expected to be logical X (flips |0><->|1> each application - this is DETERMINISTIC, trackable,
and per the paper text does not count as a stabilization failure). Test with envelope fidelity
max(F(L0),F(L1)) for round_p / alternating protocols, on a MIXED x+p displacement error."""
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

L0 = logical(0); L1 = logical(1); L0dm=ket2dm(L0); L1dm=ket2dm(L1)
gq = basis(2,0); gqdm = ket2dm(gq)
def CDgen(vhat, sigma): return (1j*tensor(vhat, sigma)).expm(method='dense')

def round_x():
    small = CDgen(k*alpha*Delta**2*pO, sigmay())
    big = CDgen(k*2*alpha*xO, sigmax())
    return small*big*small

def round_p_literal():
    # user's literal dual: CD(i*lambda,sigma_x)*CD(2*alpha,sigma_y)*CD(i*lambda,sigma_x)
    # beta=i*lam (imag): vhat = Im(i*lam)*x = lam*x ;  beta=2*alpha (real): vhat = -2*alpha*p
    small = CDgen(k*lam*xO, sigmax())
    big   = CDgen(-k*2*alpha*pO, sigmay())
    return small*big*small

Rx = round_x(); Rp = round_p_literal()

def run_alt(disp, nrounds=10, order='xp'):
    psi0 = (displace(Ncav, disp)*L0).unit()
    rho = tensor(ket2dm(psi0), gqdm)
    ro0 = rho.ptrace(0)
    hist = [(np.real(expect(L0dm,ro0)), np.real(expect(L1dm,ro0)), 1-np.real((ro0*ro0).tr()))]
    for n in range(nrounds):
        R = Rx if (order[n%len(order)]=='x') else Rp
        rho = R*rho*R.dag()
        ro = rho.ptrace(0)
        rho = tensor(ro, gqdm)
        hist.append((np.real(expect(L0dm,ro)), np.real(expect(L1dm,ro)), 1-np.real((ro*ro).tr())))
    return hist

print("=== mixed error 0.15+0.12j, alternate x,p rounds ===")
for order in ['xp','px','xx','pp']:
    h = run_alt(0.15+0.12j, 10, order)
    env = [max(f0,f1) for f0,f1,im in h]
    print(f"order={order:4s} envelope F={[round(x,4) for x in env]}")
    print(f"           impurity  ={[f'{im:.2e}' for f0,f1,im in h]}")

print("\n=== extended to 30 rounds, order=xp, disp=0.15+0.12j ===")
h = run_alt(0.15+0.12j, 30, 'xp')
env = [max(f0,f1) for f0,f1,im in h]
print("envelope F:", [round(x,4) for x in env])
print("impurity  :", [f"{im:.2e}" for f0,f1,im in h])

print("\n=== smaller displacement 0.08+0.06j, order=xp, 10 rounds ===")
h = run_alt(0.08+0.06j, 10, 'xp')
env = [max(f0,f1) for f0,f1,im in h]
print("envelope F:", [round(x,4) for x in env])
print("impurity  :", [f"{im:.2e}" for f0,f1,im in h])

print("\n=== even smaller 0.05+0.04j, order=xp, 10 rounds ===")
h = run_alt(0.05+0.04j, 10, 'xp')
env = [max(f0,f1) for f0,f1,im in h]
print("envelope F:", [round(x,4) for x in env])
print("impurity  :", [f"{im:.2e}" for f0,f1,im in h])

print("\n=== small displacement 0.03+0.025j, order=xp, 10 rounds ===")
h = run_alt(0.03+0.025j, 10, 'xp')
env = [max(f0,f1) for f0,f1,im in h]
print("envelope F:", [round(x,5) for x in env])
print("impurity  :", [f"{im:.2e}" for f0,f1,im in h])

print("\n=== disp=0.15+0.12j, 40 rounds, checking long-term plateau ===")
h = run_alt(0.15+0.12j, 40, 'xp')
env = [max(f0,f1) for f0,f1,im in h]
print("envelope F (every 4th):", [round(env[i],4) for i in range(0,len(env),4)])
print("impurity (every 4th)  :", [f"{h[i][2]:.2e}" for i in range(0,len(h),4)])
