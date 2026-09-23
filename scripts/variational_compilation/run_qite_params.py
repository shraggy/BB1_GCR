"""Re-derive the QITE-optimized DETERMINISTIC BB1(GCR) corrections (8 params: c_i, nu_i for the 4
momentum corrections; BB1 position kicks fixed), seeded from the sigma_y GCR values. Objective:
read logical-0 (eps=0) as +1 and logical-1 (eps=2u) as -1. Saves Paper_Data/qite_det_params.npy."""
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
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); b0=-0.65; b1=b0+np.sqrt(np.pi)/np.sqrt(2)
psi0=(displace(Ncav,b0)*L0).unit(); psi1=(displace(Ncav,b1)*L0).unit()
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICK=[tensor(-1j*(th/sq)*xO,SIG(ph)).expm(method='dense') for th,ph in BB1]
def Cdet(c,nu): return tensor(-1j*c*pO,SIG(nu)).expm(method='dense')
def Pp(U,psi): return float(np.real(expect(ket2dm(py),(U*tensor(psi,gq)).ptrace(1))))
seed=np.array([v for th,ph in BB1 for v in (th*ep/sq, ph+np.pi/2)])
def Uof(p):
    U=None
    for i in range(4):
        op=KICK[i]*Cdet(p[2*i],p[2*i+1]); U=op if U is None else op*U
    return U
def obj(p): U=Uof(p); return (1-Pp(U,psi0))+Pp(U,psi1)
print(f"[{time.time()-t0:.0f}s] seed obj={obj(seed):.3e}",flush=True)
res=minimize(obj,seed,method='Powell',callback=lambda xk:print(f'[{time.time()-t0:.0f}s] iter obj={obj(xk):.3e}',flush=True), options={'maxiter':60,'xtol':1e-4,'ftol':1e-8})
U=Uof(res.x); print(f"[{time.time()-t0:.0f}s] optimized: err0={1-Pp(U,psi0):.3e} err1={Pp(U,psi1):.3e} obj={res.fun:.3e}",flush=True)
np.save('Paper_Data/qite_det_params.npy',res.x)
print(f"[{time.time()-t0:.0f}s] saved params: {np.round(res.x,4)}",flush=True)
