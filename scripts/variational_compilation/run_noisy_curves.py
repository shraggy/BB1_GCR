"""Noisy readout error P(e|eps) across the Fig-3 ranges (panel a: eps in [0,u]; panel b: [2u,3u]),
amplitude-weighted gate times. Schemes: single-GCR, bare BB1, BB1(GCR) heralded, BB1(GCR) ideal.
Solid = with noise (all channels), dotted = noiseless. Shows how effective GCR is under decoherence."""
import sys, time, numpy as np
from qutip import *
t0=time.time(); Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 100
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2); I2=qeye(2); nvec=np.arange(Ncav)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); u=np.sqrt(np.pi)/(2*np.sqrt(2)); shift=np.sqrt(np.pi/2)/2
def AD(t): p=1-np.exp(-gam*t); return [tensor(qeye(Ncav),Qobj([[1,0],[0,np.sqrt(1-p)]])),tensor(qeye(Ncav),Qobj([[0,np.sqrt(p)],[0,0]]))]
SZ=tensor(qeye(Ncav),sigmaz())
def cavK(t): return [tensor(Qobj(np.diag(np.exp(-kappa*t/2*nvec))),I2), tensor(np.sqrt(1-np.exp(-kappa*t))*aC,I2)]
def noise(rho,t):
    A=AD(t); rho=A[0]*rho*A[0].dag()+A[1]*rho*A[1].dag()
    lam=1-np.exp(-gamphi*t); rho=(1-lam/2)*rho+(lam/2)*SZ*rho*SZ
    K=cavK(t); rho=K[0]*rho*K[0].dag()+K[1]*rho*K[1].dag(); return rho
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICK=[tensor(-1j*(th/sq)*xO,SIG(ph)).expm(method='dense') for th,ph in BB1]; tK=[th/np.sqrt(2*np.pi) for th,ph in BB1]
Cid=[tensor((th*ep/sq)*pO,SIG(ph)).expm(method='dense') for th,ph in BB1]
Cun=[tensor(-1j*(th*ep/sq)*pO,SIG(ph+np.pi/2)).expm(method='dense') for th,ph in BB1]
tC=[abs(th*ep/sq)/np.sqrt(2) for th,ph in BB1]
# gate lists: (op, time, nonunitary?)
SCHEMES={
 'gcr':      [(Cun[3],tC[3],False),(KICK[3],tK[3],False)],
 'bb1':      [(KICK[i],tK[i],False) for i in range(4)],
 'herald':   sum([[ (Cid[i],tC[i],True),(KICK[i],tK[i],False)] if i<3 else [(Cun[3],tC[3],False),(KICK[3],tK[3],False)] for i in range(4)],[]),
 'ideal':    sum([[(Cid[i],tC[i],True),(KICK[i],tK[i],False)] for i in range(4)],[]),
}
def Pp(scheme,beta,noisy):
    rho=tensor(ket2dm((displace(Ncav,beta)*L0).unit()),ket2dm(g))
    for op,t,nonu in SCHEMES[scheme]:
        rho=op*rho*op.dag()
        if nonu: rho=rho/rho.tr()
        if noisy: rho=noise(rho,t)
    return float(np.real(expect(ket2dm(py),rho.ptrace(1))))
epsA=np.linspace(0.0,u,16); epsB=np.linspace(2*u,3*u,16)
DATA={}
for sc in SCHEMES:
    for tag,noisy in [('n0',False),('nz',True)]:
        eA=np.array([1-Pp(sc,e-shift,noisy) for e in epsA])   # logical-0 region: err=1-P(+1)
        eB=np.array([Pp(sc,e-shift,noisy)   for e in epsB])   # logical-1 region: err=P(+1)
        DATA[f'{sc}_{tag}_a']=eA; DATA[f'{sc}_{tag}_b']=eB
    print(f"[{time.time()-t0:.0f}s] {sc}: eps=0 noiseless={1-Pp(sc,0-shift,False):.2e} noisy={1-Pp(sc,0-shift,True):.2e}",flush=True)
ov=abs((logical(0).dag()*logical(1))[0,0]); hel=0.5*(1-np.sqrt(1-ov**2))
np.savez("Paper_Data/noisy_curves.npz",epsA=epsA,epsB=epsB,hel=hel,u=u,**DATA)
print(f"[{time.time()-t0:.0f}s] saved data, Helstrom={hel:.2e}",flush=True)
