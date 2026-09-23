"""Deterministic (QITE-style) readout compilation, CORRECT convention: the readout reads the DISPLACEMENT
of a codeword -> logical-0 at eps=0 (operating point beta0), logical-1 at eps=2u=sqrt(pi/2). Ansatz keeps
the BB1(GCR) position kicks fixed and replaces the non-unitary corrections with DETERMINISTIC momentum
corrections Cdet(c,nu)=exp(-i c p sigma_nu); we optimize {c_i,nu_i}, seeded from the sigma_y GCR values,
then GROW extra deterministic correction layers. Objective: err0+err1 with err0=1-P(+1|psi0), err1=P(+1|psi1).
Baselines: single-GCR, deterministic-BB1(GCR) seed; targets: ideal/herald 5.7e-4, Helstrom ~1.1e-6.
"""
import sys, time, numpy as np
from qutip import *
from scipy.optimize import minimize
t0=time.time(); rng=np.random.default_rng(0)
def log(*a): print(*a,flush=True)
Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 40
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Delta=0.34; r=-np.log(Delta); ep=Delta**2; sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0)
b0=-0.65; b1=b0+np.sqrt(np.pi)/np.sqrt(2)   # eps=0 and eps=2u (logical-1) via x-shift sqrt(pi) -> beta=sqrt(pi)/sqrt2? use x=sqrt(pi): beta=sqrt(pi)/sqrt(2)
psi0=(displace(Ncav,b0)*L0).unit(); psi1=(displace(Ncav,b1)*L0).unit()
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICK=[tensor(-1j*(th/sq)*xO,SIG(ph)).expm(method='dense') for th,ph in BB1]
def Cdet(c,nu): return tensor(-1j*c*pO,SIG(nu)).expm(method='dense')
def Pp(U,psi): return float(np.real(expect(ket2dm(py),(U*tensor(psi,gq)).ptrace(1))))
# seed params: sigma_y GCR corrections (c_i = th_i*ep/sqrt(pi), nu_i = phi_i+pi/2)
seed=[]
for th,ph in BB1: seed += [th*ep/sq, ph+np.pi/2]
seed=np.array(seed)
def Uof(params, nlayer=4):   # first 4 corrections = BB1 pulses; extra layers appended before target
    U=None
    for i in range(nlayer):
        c,nu=params[2*i],params[2*i+1]
        if i<4: op=KICK[i]*Cdet(c,nu)
        else:   op=Cdet(c,nu)   # extra deterministic layer (no kick)
        U=op if U is None else op*U
    return U
def obj(params,nlayer=4):
    U=Uof(params,nlayer); return (1-Pp(U,psi0))+Pp(U,psi1)
# baselines
Us=Uof(seed,4); log(f"[{time.time()-t0:.0f}s] det-BB1(GCR) seed: err0={1-Pp(Us,psi0):.3e} err1={Pp(Us,psi1):.3e} obj={obj(seed):.3e}")
tgt=BB1[-1]; Ug=KICK[3]*Cdet(tgt[0]*ep/sq,tgt[1]+np.pi/2)
log(f"[{time.time()-t0:.0f}s] single-GCR (target only): err0={1-Pp(Ug,psi0):.3e} err1={Pp(Ug,psi1):.3e}")
# optimize the 4 deterministic corrections (8 params), seeded
res=minimize(lambda p:obj(p,4),seed,method='Powell',options={'maxiter':4000,'xtol':1e-5,'ftol':1e-10})
Uo=Uof(res.x,4); log(f"[{time.time()-t0:.0f}s] optimized det-BB1(GCR): err0={1-Pp(Uo,psi0):.3e} err1={Pp(Uo,psi1):.3e} obj={res.fun:.3e}")
# grow extra deterministic correction layers
cur=res.x
for extra in range(1,4):
    x0=np.concatenate([cur,[0.1, rng.uniform(0,2*np.pi)]])
    r2=minimize(lambda p:obj(p,4+extra),x0,method='Powell',options={'maxiter':4000,'xtol':1e-5,'ftol':1e-10})
    cur=r2.x; U=Uof(cur,4+extra)
    log(f"[{time.time()-t0:.0f}s] +{extra} layer(s) ({4+extra} corr): err0={1-Pp(U,psi0):.3e} err1={Pp(U,psi1):.3e} obj={r2.fun:.3e}")
log(f"[{time.time()-t0:.0f}s] DONE (herald floor 5.7e-4)")
