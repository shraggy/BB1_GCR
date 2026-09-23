"""Gate-level Kraus noise simulation of ONE RUS attempt = the shallow heralded BB1(GCR) readout
(3 heralded corrections + deterministic GCR target), on the full GKP logical-0 state at eps=0.
Density matrix on oscillator (x) transmon (x) herald-ancilla. After each conditional displacement we
apply decoherence for t_CD; heralds are post-selected (project ancilla=0). We isolate each channel
(ancilla decay/dephasing, transmon decay/dephasing, cavity photon loss) and the total, and report the
accepted-shot readout infidelity 1-P(+1) at eps=0 (noiseless floor ~5.7e-4).
Rates (Sivak/manuscript): gamma=gamma_phi=1/200 us^-1 (T1=Tphi=200us), kappa=1/1000 us^-1; t_CD=1 us.
"""
import sys, time, numpy as np
from qutip import *
t0=time.time()
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 40
tCD  = float(sys.argv[2]) if len(sys.argv)>2 else 1.0   # us per conditional displacement
gam, gamphi, kappa = 1/200., 1/200., 1/1000.            # us^-1
# --- operators on osc (x) transmon (x) ancilla ---
Ic,I2 = qeye(Ncav), qeye(2)
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
g=basis(2,0); e=basis(2,1); py=(g+1j*e).unit()
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
# gates (3-subsystem):
def Opx(th,phi):  return tensor(-1j*(th/sq)*xO, SIG(phi), I2).expm()                # position kick (osc,transmon)
def Wgad(th,phi): return tensor(-1j*(th*ep/sq)*pO, SIG(phi), sigmay()).expm()        # ancilla-controlled corr
Had = tensor(Ic, I2, (1/np.sqrt(2))*Qobj([[1,1],[1,-1]]))                            # Hadamard on ancilla
def Opy_det(th,phi): return tensor(-1j*(th*ep/sq)*pO, SIG(phi+np.pi/2), I2).expm()   # deterministic GCR target
P0a = tensor(Ic, I2, ket2dm(g))                                                      # herald projector (ancilla=0)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
# --- decoherence Kraus for duration tCD, per channel ---
def qubit_kraus(slot, chan):  # slot: 1=transmon,2=ancilla ; chan: 'decay'|'deph'
    ops=[Ic,I2,I2]
    if chan=='decay':
        p=1-np.exp(-gam*tCD); M0=Qobj([[1,0],[0,np.sqrt(1-p)]]); M1=Qobj([[0,np.sqrt(p)],[0,0]])
        Ks=[M0,M1]
    else:
        lam=1-np.exp(-gamphi*tCD); M0=np.sqrt(1-lam/2)*I2; M1=np.sqrt(lam/2)*sigmaz(); Ks=[M0,M1]
    out=[]
    for K in Ks:
        o=ops.copy(); o[slot]=K; out.append(tensor(o))
    return out
def cav_kraus():  # photon loss for tCD (leading no-jump + single-jump)
    eta=np.exp(-kappa*tCD); K0=tensor((-kappa*tCD/2*aC.dag()*aC).expm(),I2,I2)
    K1=tensor(np.sqrt(1-eta)*aC,I2,I2); return [K0,K1]
KRAUS={'anc_decay':qubit_kraus(2,'decay'),'anc_deph':qubit_kraus(2,'deph'),
       'tr_decay':qubit_kraus(1,'decay'),'tr_deph':qubit_kraus(1,'deph'),'cav_loss':cav_kraus()}
def noise(rho,active):
    for ch in active:
        rho=sum(K*rho*K.dag() for K in KRAUS[ch])
    return rho
def run(active):
    a=-np.sqrt(np.pi)/2; psi_in=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
    rho=tensor(ket2dm(psi_in),ket2dm(g),ket2dm(g))
    for i,(th,phi) in enumerate(BB1):
        if i<3:  # heralded correction
            U=Wgad(th,phi); rho=U*rho*U.dag(); rho=noise(rho,active)
            rho=Had*rho*Had.dag()
            rho=P0a*rho*P0a; rho=rho/rho.tr()                 # herald: post-select ancilla=0
        else:    # deterministic target
            U=Opy_det(th,phi); rho=U*rho*U.dag(); rho=noise(rho,active)
        K=Opx(th,phi); rho=K*rho*K.dag(); rho=noise(rho,active)
    rq=rho.ptrace(1)
    return 1-float(np.real(expect(ket2dm(py),rq)))            # readout infidelity 1-P(+1)
a=-np.sqrt(np.pi)/2; _pk=(displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav}, tCD={tCD}us, peak <n>={float(expect(num(Ncav),_pk)):.2f}")
for name,active in [('noiseless',[]),('anc_decay',['anc_decay']),('anc_deph',['anc_deph']),
                    ('tr_decay',['tr_decay']),('tr_deph',['tr_deph']),('cav_loss',['cav_loss']),
                    ('TOTAL',['anc_decay','anc_deph','tr_decay','tr_deph','cav_loss'])]:
    print(f"[{time.time()-t0:.0f}s] {name:10s} readout infidelity = {run(active):.4e}")
