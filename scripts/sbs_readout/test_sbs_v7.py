"""Test round_p with the GCR-generalization Delta -> 1/Delta substitution for the small kick
(paper: 'the correction should act along p rather than x... achieved by choosing v=p, replacing
Delta by 1/Delta')."""
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

def round_p(sign_small=1, sign_big=1, sig_small='y', sig_big='x'):
    SIG = {'x':sigmax(),'y':sigmay()}
    small = CDgen(sign_small*k*alpha*(1/Delta)**2*xO, SIG[sig_small])
    big   = CDgen(sign_big*k*2*alpha*pO, SIG[sig_big])
    return small*big*small

def run(R, disp, nrounds=10):
    psi0 = (displace(Ncav, disp)*L0).unit()
    rho = tensor(ket2dm(psi0), gqdm)
    fids=[float(np.real(expect(L0dm, rho.ptrace(0))))]
    impurs=[1-float(np.real((rho.ptrace(0)*rho.ptrace(0)).tr()))]
    for n in range(nrounds):
        rho = R*rho*R.dag()
        ro = rho.ptrace(0)
        rho = tensor(ro, gqdm)
        fids.append(float(np.real(expect(L0dm, ro))))
        impurs.append(1-float(np.real((ro*ro).tr())))
    return fids, impurs

print("round_x sanity, pure-x-error 0.2:", [round(f,4) for f in run(round_x(),0.2)[0]])
print()
for sig_small in ['x','y']:
    for sig_big in ['x','y']:
        for ss in [1,-1]:
            for sb in [1,-1]:
                f,im = run(round_p(ss,sb,sig_small,sig_big), 0.2j, 10)
                tag = "CLEAN" if (f[-1]>0.9 and all(np.diff(f) > -0.05)) else "     "
                print(f"{tag} small_sig={sig_small} big_sig={sig_big} ss={ss:+d} sb={sb:+d}  F={[round(x,3) for x in f]}")
