"""RUS exact-M noise budget with AMPLITUDE-DEPENDENT gate times: a conditional displacement of
amplitude |beta| takes |beta| microseconds (amplitude-1 CD = 1 us); decoherence prob per gate =
1-exp(-rate*|beta|*1us), rates fixed (T1=Tphi=200us, cavity T1=1000us). Position kick e^{-i(th/sqrt(pi))x sig}
has |beta|=th/sqrt(2 pi); momentum correction e^{-i c p sig} has |beta|=c/sqrt2, c=th*ep/sqrt(pi).
Noise applied after each gate for that gate's duration. Exact non-unitary M (renormalize=herald success),
full GKP logical-0 at operating point. Isolate transmon decay/dephasing + cavity loss + total.
"""
import sys, time, numpy as np
from qutip import *
t0=time.time(); Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 100
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2); I2=qeye(2)
nvec=np.arange(Ncav)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
CORR=[]; KICK=[]; tKICK=[]; tCORR=[]
for i,(th,phi) in enumerate(BB1):
    KICK.append(tensor(-1j*(th/sq)*xO,SIG(phi)).expm(method='dense')); tKICK.append(th/np.sqrt(2*np.pi))
    c=th*ep/sq; tCORR.append(abs(c)/np.sqrt(2))
    if i<3: CORR.append(tensor(c*pO,SIG(phi)).expm(method='dense'))
    else:   CORR.append(tensor(-1j*c*pO,SIG(phi+np.pi/2)).expm(method='dense'))
Ttot=sum(tKICK)+sum(tCORR)
print(f"[{time.time()-t0:.0f}s] kick times={[round(x,2) for x in tKICK]} corr times={[round(x,3) for x in tCORR]} total={Ttot:.2f} us")
# amplitude/time-dependent channels
def AD(t):
    p=1-np.exp(-gam*t); return [tensor(qeye(Ncav),Qobj([[1,0],[0,np.sqrt(1-p)]])),tensor(qeye(Ncav),Qobj([[0,np.sqrt(p)],[0,0]]))]
SZ=tensor(qeye(Ncav),sigmaz())
def cavK(t):
    K0=tensor(Qobj(np.diag(np.exp(-kappa*t/2*nvec))),I2); K1=tensor(np.sqrt(1-np.exp(-kappa*t))*aC,I2); return [K0,K1]
def noise(rho,ch,t):
    if 'tr_decay' in ch:
        A=AD(t); rho=A[0]*rho*A[0].dag()+A[1]*rho*A[1].dag()
    if 'tr_deph' in ch:
        lam=1-np.exp(-gamphi*t); rho=(1-lam/2)*rho+(lam/2)*SZ*rho*SZ
    if 'cav_loss' in ch:
        K=cavK(t); rho=K[0]*rho*K[0].dag()+K[1]*rho*K[1].dag()
    return rho
tgt=logical(0); b0=-0.65
def readout(ch):
    rho=tensor(ket2dm((displace(Ncav,b0)*tgt).unit()),ket2dm(g))
    for i in range(4):
        C=CORR[i]; rho=C*rho*C.dag()
        if i<3: rho=rho/rho.tr()
        if ch: rho=noise(rho,ch,tCORR[i])
        K=KICK[i]; rho=K*rho*K.dag()
        if ch: rho=noise(rho,ch,tKICK[i])
    return 1-float(np.real(expect(ket2dm(py),rho.ptrace(1))))
res={}
for name,ch in [('noiseless',[]),('tr_decay',['tr_decay']),('tr_deph',['tr_deph']),
                ('cav_loss',['cav_loss']),('TOTAL',['tr_decay','tr_deph','cav_loss'])]:
    res[name]=readout(ch); print(f"[{time.time()-t0:.0f}s] {name:10s}={res[name]:.4e}  (+{res[name]-res['noiseless']:.2e})",flush=True)
