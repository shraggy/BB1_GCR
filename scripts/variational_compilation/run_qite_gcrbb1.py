"""Can QITE improve the SINGLE small corrective conditional displacement placed BEFORE the four bare
BB1 kicks (the GCR-BB1 construction, Initial_repo cell 54)? Structure:
    Cdet(c,nu) = exp(-i c pO SIG(nu))   -- ONE pre-correction, then 4 BARE BB1 kicks (no per-kick corr).
Analytic value (cell 54): gate = exp(+i (beta*ep/4) pO sigma_y) => c0 = -(beta*ep/4), nu0 = pi/2.
Optimize the 2 params (c,nu) via Powell, seeded from the analytic value. Objective (same as the
BB1(GCR) QITE): read logical-0 (eps=0) as +1 and logical-1 (eps=2u) as -1.
Saves Paper_Data/qite_gcrbb1_params.npy."""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np, time
from qutip import *
from scipy.optimize import minimize
t0=time.time(); Ncav=80
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Delta=0.34; r=-np.log(Delta); ep=Delta**2; sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
beta=(np.pi/2)/2/(np.sqrt(np.pi)/2)   # = sqrt(pi)/2, the cell-54 beta
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); b0=-0.65; b1=b0+np.sqrt(np.pi)/np.sqrt(2)
psi0=(displace(Ncav,b0)*L0).unit(); psi1=(displace(Ncav,b1)*L0).unit()
# four BARE BB1 kicks (no per-kick momentum correction) -- the defining feature of GCR-BB1
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICK=[tensor(-1j*(th/sq)*xO,SIG(ph)).expm(method='dense') for th,ph in BB1]
BB1U=None
for K in KICK: BB1U = K if BB1U is None else K*BB1U     # Op4*Op3*Op2*Op1 applied left-to-right
def Cdet(c,nu): return tensor(-1j*c*pO,SIG(nu)).expm(method='dense')
def Pp(U,psi): return float(np.real(expect(ket2dm(py),(U*tensor(psi,gq)).ptrace(1))))
def Uof(p): return BB1U*Cdet(p[0],p[1])                 # pre-correction FIRST, then bare BB1
seed=np.array([-(beta*ep/4), np.pi/2])                  # analytic cell-54 value
def obj(p): U=Uof(p); return (1-Pp(U,psi0))+Pp(U,psi1)
Us=Uof(seed); print(f"[{time.time()-t0:.0f}s] seed c={seed[0]:+.4f} nu={seed[1]:.4f}  "
      f"err0={1-Pp(Us,psi0):.3e} err1={Pp(Us,psi1):.3e} obj={obj(seed):.3e}",flush=True)
res=minimize(obj,seed,method='Powell',
    callback=lambda xk:print(f'[{time.time()-t0:.0f}s] iter obj={obj(xk):.3e} p={np.round(xk,4)}',flush=True),
    options={'maxiter':200,'xtol':1e-5,'ftol':1e-10})
U=Uof(res.x)
print(f"[{time.time()-t0:.0f}s] OPT c={res.x[0]:+.4f} nu={res.x[1]:.4f}  "
      f"err0={1-Pp(U,psi0):.3e} err1={Pp(U,psi1):.3e} obj={res.fun:.3e}",flush=True)
np.save('Paper_Data/qite_gcrbb1_params.npy',res.x)
print(f"[{time.time()-t0:.0f}s] saved {np.round(res.x,5)}",flush=True)
