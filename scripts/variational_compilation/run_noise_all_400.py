"""All readout noise budgets at Ncav=400 (paper-grade), amplitude-weighted gate times (|beta| us).
single-GCR, bare BB1, and heralded/RUS BB1(GCR) (exact non-unitary M, renormalize=herald success).
Channels: transmon decay/dephasing + cavity loss. Rates T1=Tphi=200us, cavity 1000us."""
import sys, time, numpy as np
from qutip import *
t0=time.time(); Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 400
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
psi0=(displace(Ncav,-0.65)*logical(0)).unit()
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} built, <n>={float(expect(num(Ncav),logical(0))):.2f}",flush=True)
def AD(t): p=1-np.exp(-gam*t); return [tensor(qeye(Ncav),Qobj([[1,0],[0,np.sqrt(1-p)]])),tensor(qeye(Ncav),Qobj([[0,np.sqrt(p)],[0,0]]))]
SZ=tensor(qeye(Ncav),sigmaz())
def cavK(t): return [tensor(Qobj(np.diag(np.exp(-kappa*t/2*nvec))),I2), tensor(np.sqrt(1-np.exp(-kappa*t))*aC,I2)]
def noise(rho,ch,t):
    if 'd' in ch: A=AD(t); rho=A[0]*rho*A[0].dag()+A[1]*rho*A[1].dag()
    if 'p' in ch: lam=1-np.exp(-gamphi*t); rho=(1-lam/2)*rho+(lam/2)*SZ*rho*SZ
    if 'c' in ch: K=cavK(t); rho=K[0]*rho*K[0].dag()+K[1]*rho*K[1].dag()
    return rho
Pm=lambda U:1-float(np.real(expect(ket2dm(py),(U).ptrace(1))))
def budget(label, gates, times, corr_nonunit):
    def run(ch):
        rho=tensor(ket2dm(psi0),ket2dm(g))
        for k,(G,t) in enumerate(zip(gates,times)):
            rho=G*rho*G.dag()
            if corr_nonunit[k]: rho=rho/rho.tr()
            if ch: rho=noise(rho,ch,t)
        return 1-float(np.real(expect(ket2dm(py),rho.ptrace(1))))
    n0=run('')
    row=" ".join(f"{nm}={run(c):.3e}" for nm,c in [('nl',''),('dec','d'),('dep','p'),('cav','c'),('TOT','dpc')])
    tot=run('dpc')
    print(f"[{time.time()-t0:.0f}s] {label:16s} T={sum(times):.2f}us floor={n0:.3e} TOTAL={tot:.3e} (+{tot-n0:.2e})  [{row}]",flush=True)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
KICK=[tensor(-1j*(th/sq)*xO,SIG(ph)).expm(method='dense') for th,ph in BB1]; tK=[th/np.sqrt(2*np.pi) for th,ph in BB1]
# single-GCR: correction(sigma_y) then kick, target pulse th=pi/2
th=np.pi/2; c=th*ep/sq
budget("single-GCR",[tensor(-1j*c*pO,SIG(np.pi/2)).expm(method='dense'),KICK[3]],[abs(c)/np.sqrt(2),tK[3]],[False,False])
# bare BB1: 4 kicks only
budget("bare-BB1",KICK,tK,[False]*4)
# heralded BB1(GCR): exact M (nonunit) for i<3 + det target, interleaved with kicks
gates=[]; times=[]; nonu=[]
for i,(th,ph) in enumerate(BB1):
    c=th*ep/sq
    if i<3: gates.append(tensor(c*pO,SIG(ph)).expm(method='dense')); nonu.append(True)
    else:   gates.append(tensor(-1j*c*pO,SIG(ph+np.pi/2)).expm(method='dense')); nonu.append(False)
    times.append(abs(c)/np.sqrt(2))
    gates.append(KICK[i]); times.append(tK[i]); nonu.append(False)
budget("BB1(GCR)-herald",gates,times,nonu)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
