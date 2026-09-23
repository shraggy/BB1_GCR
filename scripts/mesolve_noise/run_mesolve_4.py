"""mesolve readout-error curves across the Fig-3 ranges (panel a: eps in [0,u]; b: [2u,3u]) at Ncav=400.
Four cases: GCR (single), BB1(GCR) deterministic-from-QITE, bare BB1 (readouts); heralded BB1(GCR)+RUS
(calibration). Five noise conditions: noiseless, complete, transmon decay, transmon dephasing, oscillator
decay. (No oscillator dephasing -- not in the Sivak model.) Continuous master-equation evolution: each
UNITARY gate g=exp(-i K) of duration t is evolved under H=K/t with c_ops for time t (CD amplitude |beta|
sets t via |beta| us). The heralded case's 3 non-unitary corrections are applied instantaneously as the
exact M (renormalize=herald success), since they are short; its kicks + det target are mesolve'd.
Rates: gamma=gamma_phi=1/200, kappa=1/1000 us^-1."""
import sys, time, numpy as np
from qutip import *
t0=time.time(); Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 400
NE=int(sys.argv[2]) if len(sys.argv)>2 else 8   # eps points per panel
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); u=np.sqrt(np.pi)/(2*np.sqrt(2)); shift=np.sqrt(np.pi/2)/2
qp=np.load('Paper_Data/qite_det_params.npy')   # QITE-optimized c_i,nu_i (4 corrections)
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav}, QITE params loaded: {np.round(qp,3)}",flush=True)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
# generator K (gate=exp(-iK)) and duration t=|beta|
def kick(i): th,ph=BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph=BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')  # non-unitary
def U(K): return (-1j*K).expm(method='dense')
# build schemes as lists of (kind, op, K, t): kind 'U'=unitary(mesolve), 'M'=instantaneous exact-M
def sc_gcr():
    K,t=det_corr((np.pi/2)*ep/sq, 0+np.pi/2); Kk,tk=kick(3)
    return [('U',U(K),K,t),('U',U(Kk),Kk,tk)]
def sc_bb1():
    L=[]
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_qite():
    L=[]
    for i in range(4):
        K,t=det_corr(qp[2*i],qp[2*i+1]); Kk,tk=kick(i)
        L+=[('U',U(K),K,t),('U',U(Kk),Kk,tk)]
    return L
def sc_herald():
    L=[]
    for i in range(4):
        Kk,tk=kick(i)
        if i<3: L.append(('M',Mexact(i),None,0.0))
        else:   K,t=det_corr((np.pi/2)*ep/sq,0+np.pi/2); L.append(('U',U(K),K,t))
        L.append(('U',U(Kk),Kk,tk))
    return L
SCHEMES={'GCR':sc_gcr(),'BB1(GCR)-QITE':sc_qite(),'BB1':sc_bb1(),'herald-RUS':sc_herald()}
print(f"[{time.time()-t0:.0f}s] schemes built",flush=True)
sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)
def readout(scheme, beta, cond):
    psi=(displace(Ncav,beta)*L0).unit(); pdm=ket2dm(psi)
    rho=tensor(pdm,ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in scheme:
        if kind=='M': rho=op*rho*op.dag(); rho=rho/rho.tr()
        elif cops is None: rho=op*rho*op.dag()
        else: rho=mesolve(K/t,rho,[0,t],c_ops=cops,options=opts).states[-1]
    qerr=1-float(np.real(expect(pydm,rho.ptrace(1))))                 # readout()=1-P(+1)
    ro=rho.ptrace(0)
    impur=1-float(np.real((ro*ro).tr()))                             # oscillator impurity = decoherence damage
    oraw=1-float(np.real(expect(pdm,ro)))                             # infidelity vs start (dominated by coherent readout shift)
    dx=float(np.real(expect(xO,ro)))-float(np.real(expect(xO,psi)))
    dp=float(np.real(expect(pO,ro)))-float(np.real(expect(pO,psi)))
    D=displace(Ncav,-(dx+1j*dp)/np.sqrt(2))
    ocorr=1-float(np.real(expect(pdm,D*ro*D.dag())))                  # mean-corrected infidelity
    return qerr,impur,oraw,ocorr
epsA=np.linspace(0,u,NE); epsB=np.linspace(2*u,3*u,NE)
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
DATA={'epsA':epsA,'epsB':epsB,'u':u}
for scn,scheme in SCHEMES.items():
    for cond in conds:
        rA=np.array([readout(scheme,e-shift,cond) for e in epsA])   # cols: qerr, impurity, oraw, ocorr
        rB=np.array([readout(scheme,e-shift,cond) for e in epsB])
        DATA[f'{scn}|{cond}|a|q']=rA[:,0];     DATA[f'{scn}|{cond}|b|q']=1-rB[:,0]
        DATA[f'{scn}|{cond}|a|impur']=rA[:,1]; DATA[f'{scn}|{cond}|b|impur']=rB[:,1]
        DATA[f'{scn}|{cond}|a|oraw']=rA[:,2];  DATA[f'{scn}|{cond}|b|oraw']=rB[:,2]
        DATA[f'{scn}|{cond}|a|ocorr']=rA[:,3]; DATA[f'{scn}|{cond}|b|ocorr']=rB[:,3]
        print(f"[{time.time()-t0:.0f}s] {scn:14s} {cond:10s} q@eps0={rA[0,0]:.3e} oscImpurity@eps0={rA[0,1]:.3e}",flush=True)
    np.savez('Paper_Data/mesolve_4.npz',**DATA)
print(f"[{time.time()-t0:.0f}s] DONE, data saved",flush=True)
