"""Fixed test-disp pairing: big_quad='x' -> test disp=0.2 (real x-error, correctable by round_x);
big_quad='p' -> test disp=0.2j (imaginary p-error, should be correctable by a true round_p)."""
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

QUAD = {'x': xO, 'p': pO}
SIG = {'x': sigmax(), 'y': sigmay()}

def build_round(big_quad, big_sig, big_sign, small_quad, small_sig, small_sign):
    big = CDgen(big_sign*k*2*alpha*QUAD[big_quad], SIG[big_sig])
    small = CDgen(small_sign*k*alpha*Delta**2*QUAD[small_quad], SIG[small_sig])
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

def is_clean_monotonic(fids):
    arr = np.array(fids)
    diffs = np.diff(arr)
    n_big_drop = np.sum(diffs < -0.05)
    return n_big_drop == 0 and arr[-1] > 0.9 and arr[-1] > arr[0]

print("=== big_quad='p' (searching for true round_p), test disp=0.2j ===")
winners=[]
for small_quad in ['x']:  # perpendicular to p is x
    for big_sig in ['x','y']:
        for small_sig in ['x','y']:
            for big_sign in [1,-1]:
                for small_sign in [1,-1]:
                    R = build_round('p',big_sig,big_sign, small_quad,small_sig,small_sign)
                    fids = run(R, 0.2j, 10)
                    ok = is_clean_monotonic(fids)
                    tag = "CLEAN" if ok else "     "
                    print(f"{tag} big=(p,{big_sig},{big_sign:+d}) small=({small_quad},{small_sig},{small_sign:+d})  F={[round(x,3) for x in fids]}")
                    if ok: winners.append((big_sig,big_sign,small_sig,small_sign))
print(f"\n{len(winners)} clean winners for round_p:", winners)
