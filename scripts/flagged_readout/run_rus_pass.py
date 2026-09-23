"""Noise-reduced herald pass rate + average retries for RUS (3-herald).
Track the UN-normalized density matrix through the noisy circuit (apply exact non-unitary M without
renormalizing; noise channels are trace-preserving, so trace reduction = norm reduction reshaped by
system noise). Pass rate = tr(rho_final)/lambda^2, with lambda = top singular value of the full flagged
operator on the low-energy (n<=40) subspace -- exactly the bound used for the noiseless 17%. Average
retries = 1/pass. Isolate transmon decay/dephasing + cavity loss (system channels the norm sees).
"""
import sys, time, numpy as np
from qutip import *
t0=time.time()
Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 100; Nsub=40
tCD=1.0; gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2); I2=qeye(2)
def SIG(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
g=basis(2,0); e=basis(2,1); py=(g+1j*e).unit()
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
CORR=[]; KICK=[]
for i,(th,phi) in enumerate(BB1):
    KICK.append(tensor(-1j*(th/sq)*xO, SIG(phi)).expm(method='dense'))
    if i<3: CORR.append(tensor((th*ep/sq)*pO, SIG(phi)).expm(method='dense'))
    else:   CORR.append(tensor(-1j*(th*ep/sq)*pO, SIG(phi+np.pi/2)).expm(method='dense'))
# subspace bound lambda from the full flagged operator U = KICK3 CORR3 ... KICK0 CORR0
U=KICK[0]*CORR[0]
for i in range(1,4): U=KICK[i]*CORR[i]*U
Uf=U.full(); lam=np.linalg.svd(Uf[:,:2*Nsub], compute_uv=False)[0]
print(f"[{time.time()-t0:.0f}s] lambda(subspace n<={Nsub}) = {lam:.4f}")
# noise Kraus (osc(x)transmon)
p_ad=1-np.exp(-gam*tCD); lamd=1-np.exp(-gamphi*tCD)
AD0=tensor(qeye(Ncav),Qobj([[1,0],[0,np.sqrt(1-p_ad)]])); AD1=tensor(qeye(Ncav),Qobj([[0,np.sqrt(p_ad)],[0,0]]))
SZ=tensor(qeye(Ncav),sigmaz())
K0c=tensor((-kappa*tCD/2*aC.dag()*aC).expm(method='dense'),I2); K1c=tensor(np.sqrt(1-np.exp(-kappa*tCD))*aC,I2)
def noise(rho,ch):
    if 'tr_decay' in ch: rho=AD0*rho*AD0.dag()+AD1*rho*AD1.dag()
    if 'tr_deph'  in ch: rho=(1-lamd/2)*rho+(lamd/2)*SZ*rho*SZ
    if 'cav_loss' in ch: rho=K0c*rho*K0c.dag()+K1c*rho*K1c.dag()
    return rho
tgt=logical(0); b0=-0.650
def run(ch):
    rho=tensor(ket2dm((displace(Ncav,b0)*tgt).unit()),ket2dm(g))   # tr=1
    for i in range(4):
        C=CORR[i]; rho=C*rho*C.dag()            # NO renormalization (track norm)
        if ch: rho=noise(rho,ch)
        K=KICK[i]; rho=K*rho*K.dag()
        if ch: rho=noise(rho,ch)
    tr=float(np.real(rho.tr())); pas=tr/lam**2
    infid=1-float(np.real(expect(ket2dm(py),(rho/tr).ptrace(1))))
    return pas,infid
print(f"[{time.time()-t0:.0f}s] <n>={float(expect(num(Ncav),tgt)):.2f}  operating beta={b0}")
print(f"{'channel':14s} {'pass %':>8s} {'retries':>9s} {'infid':>10s}")
for name,ch in [('noiseless',[]),('tr_decay',['tr_decay']),('tr_deph',['tr_deph']),
                ('cav_loss',['cav_loss']),('ALL system',['tr_decay','tr_deph','cav_loss'])]:
    pas,infid=run(ch)
    print(f"{name:14s} {100*pas:8.2f} {1/pas:9.2f} {infid:10.3e}   [{time.time()-t0:.0f}s]")
