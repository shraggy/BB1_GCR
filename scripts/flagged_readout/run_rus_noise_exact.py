"""Option (a): EXACT RUS noise sim. Corrections use the exact non-unitary M (renormalize = herald
success) so the noiseless floor is the TRUE readout floor (~6e-4), not the first-order gadget (0.134).
Input = full GKP logical-0 codeword at its operating point (<n>~6, exact cavity term). Density matrix
on oscillator (x) transmon. Gate operators PRECOMPUTED once. Isolate transmon decay/dephasing + cavity
photon loss (ancilla channels carried from the gadget run: <0.05%, herald-protected).
Rates: gamma=gamma_phi=1/200, kappa=1/1000 us^-1; tCD=1us.
"""
import sys, time, numpy as np
from qutip import *
t0=time.time()
Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 100
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
# --- precompute gate operators ONCE ---
CORR=[]; KICK=[]
for i,(th,phi) in enumerate(BB1):
    KICK.append(tensor(-1j*(th/sq)*xO, SIG(phi)).expm(method='dense'))
    if i<3: CORR.append(tensor((th*ep/sq)*pO, SIG(phi)).expm(method='dense'))        # exact non-unitary M
    else:   CORR.append(tensor(-1j*(th*ep/sq)*pO, SIG(phi+np.pi/2)).expm(method='dense'))  # det. GCR target
print(f"[{time.time()-t0:.0f}s] gates built (Ncav={Ncav})")
# --- decoherence Kraus (osc (x) transmon), duration tCD ---
p_ad=1-np.exp(-gam*tCD); lam=1-np.exp(-gamphi*tCD)
AD0=tensor(qeye(Ncav),Qobj([[1,0],[0,np.sqrt(1-p_ad)]])); AD1=tensor(qeye(Ncav),Qobj([[0,np.sqrt(p_ad)],[0,0]]))
SZ=tensor(qeye(Ncav),sigmaz())
K0c=tensor((-kappa*tCD/2*aC.dag()*aC).expm(method='dense'),I2); K1c=tensor(np.sqrt(1-np.exp(-kappa*tCD))*aC,I2)
def noise(rho,ch):
    if 'tr_decay' in ch: rho=AD0*rho*AD0.dag()+AD1*rho*AD1.dag()
    if 'tr_deph'  in ch: rho=(1-lam/2)*rho+(lam/2)*SZ*rho*SZ
    if 'cav_loss' in ch: rho=K0c*rho*K0c.dag()+K1c*rho*K1c.dag()
    return rho
tgt=logical(0)
def readout(beta, ch):
    rho=tensor(ket2dm((displace(Ncav,beta)*tgt).unit()),ket2dm(g))
    for i in range(4):
        C=CORR[i]; rho=C*rho*C.dag()
        if i<3: rho=rho/rho.tr()          # herald success (renormalize)
        if ch: rho=noise(rho,ch)
        K=KICK[i]; rho=K*rho*K.dag()
        if ch: rho=noise(rho,ch)
    return 1-float(np.real(expect(ket2dm(py),rho.ptrace(1))))
betas=np.linspace(-0.95,-0.30,14); errs=[readout(b,[]) for b in betas]
b0=betas[int(np.argmin(errs))]
print(f"[{time.time()-t0:.0f}s] <n>={float(expect(num(Ncav),tgt)):.2f}  operating beta={b0:.3f}  floor={min(errs):.3e}")
res={}
for name,ch in [('noiseless',[]),('tr_decay',['tr_decay']),('tr_deph',['tr_deph']),
                ('cav_loss',['cav_loss']),('TOTAL',['tr_decay','tr_deph','cav_loss'])]:
    res[name]=readout(b0,ch); print(f"[{time.time()-t0:.0f}s] {name:10s} = {res[name]:.4e}   (+{res[name]-res['noiseless']:.2e})")
