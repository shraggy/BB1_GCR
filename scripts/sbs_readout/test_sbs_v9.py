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

L0 = logical(0); L1 = logical(1); L0dm=ket2dm(L0); L1dm=ket2dm(L1)
gq = basis(2,0); gqdm = ket2dm(gq)
def CDgen(vhat, sigma): return (1j*tensor(vhat, sigma)).expm(method='dense')

def round_x():
    small = CDgen(k*alpha*Delta**2*pO, sigmay())
    big = CDgen(k*2*alpha*xO, sigmax())
    return small*big*small

# check round_x on ideal L0 (should be near-invariant)
Rx = round_x()
rho = tensor(L0dm, gqdm)
rho2 = Rx*rho*Rx.dag()
ro = rho2.ptrace(0)
print("round_x on ideal L0: F(L0)=",np.real(expect(L0dm,ro))," F(L1)=",np.real(expect(L1dm,ro))," purity=",np.real((ro*ro).tr()))

QUAD={'x':xO,'p':pO}; SIG={'x':sigmax(),'y':sigmay()}
print("\nAll big=p round_p variants on ideal L0 (disp=0), Delta^2 coefficient:")
for big_sig in ['x','y']:
    for small_sig in ['x','y']:
        for sb in [1,-1]:
            for ss in [1,-1]:
                small = CDgen(ss*k*alpha*Delta**2*QUAD['x'], SIG[small_sig])
                big   = CDgen(sb*k*2*alpha*QUAD['p'], SIG[big_sig])
                R = small*big*small
                rho2 = R*rho*R.dag()
                ro = rho2.ptrace(0)
                fL0=np.real(expect(L0dm,ro)); fL1=np.real(expect(L1dm,ro)); pur=np.real((ro*ro).tr())
                print(f"big_sig={big_sig} small_sig={small_sig} sb={sb:+d} ss={ss:+d}  F(L0)={fL0:.4f} F(L1)={fL1:.4f} purity={pur:.4f}")
