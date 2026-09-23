"""Recompute ONLY the '...|infid' keys in Paper_Data/damage_readout.npz using the two-defect-fixed
definitions (see run_damage_readout.py):
  (1) D calibrated from RAW logical(0) (beta=0), not the shifted L0.
  (2) infidelity referenced to the PRISTINE canonical codeword PREF[stname] (ket2dm(L0)/ket2dm(L1)),
      not the damaged input.
All '...|err' keys and 'times'/'kappa'/'kappa_phi' are preserved EXACTLY (left as stored). The recomputed
err is used only for VALIDATION against the stored err (atol=2e-3); a mismatch aborts before any write.
Damage states (28 = 2 channels x 2 states x 7 times) are cached once (protocol-independent) so the long
idle evolution is not repeated per protocol. Ncav=400, checkpoint after each protocol."""
import os, sys, time, numpy as np
from qutip import *
t0=time.time()
Ncav=400
gam=1/200.; gamphi=1/200.; kappa=1/1000.
kappa_phi=1/5000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
shift=np.sqrt(np.pi/2)/2
L0=(displace(Ncav,-shift)*logical(0)).unit(); L1=(displace(Ncav,-shift)*logical(1)).unit()
L0raw=logical(0)   # RAW comb (beta=0); used to calibrate D (fixed defect #1)
STATES={'L0':L0,'L1':L1}
PREF={'L0':ket2dm(L0),'L1':ket2dm(L1)}   # pristine codewords (infidelity reference, fixed defect #2)
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} shift={shift:.5f} kappa={kappa} kappa_phi={kappa_phi}",flush=True)

# ---- BB1 / kick / det_corr / U / Mexact -- verbatim from run_damage_readout.py ----
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph=BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph=BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def U(K): return (-1j*K).expm(method='dense')

qp=np.load('Paper_Data/qite_det_params.npy')
gp=np.load('Paper_Data/qite_gcrbb1_params.npy')
_blk=np.load('Paper_Data/prepended_block_params.npz')
paramsA=_blk['variantA_params']; paramsB=_blk['variantB_params']

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
def sc_gcrbb1():
    L=[]
    Kc,tc=det_corr(gp[0],gp[1]); L.append(('U',U(Kc),Kc,tc))
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_block(p):
    L=[]
    for k in range(4): Kc,tc=det_corr(p[2*k],p[2*k+1]); L.append(('U',U(Kc),Kc,tc))
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_herald():
    L=[]
    for i in range(4):
        Kk,tk=kick(i)
        if i<3: L.append(('M',Mexact(i),None,0.0))
        else:   K,t=det_corr((np.pi/2)*ep/sq,0+np.pi/2); L.append(('U',U(K),K,t))
        L.append(('U',U(Kk),Kk,tk))
    return L

PROTOCOLS=['GCR','BB1(GCR)-QITE','BB1','GCR-BB1','GCR-BB1-blockA','GCR-BB1-blockB','herald-RUS']
def build_scheme(name):
    return {'GCR':sc_gcr,'BB1(GCR)-QITE':sc_qite,'BB1':sc_bb1,'GCR-BB1':sc_gcrbb1,
            'GCR-BB1-blockA':lambda:sc_block(paramsA),'GCR-BB1-blockB':lambda:sc_block(paramsB),
            'herald-RUS':sc_herald}[name]()

sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)
READ_CONDS=['noiseless','tr_deph','tr_decay','osc_decay']
CHANNELS=['osc_decay','osc_deph']
DAMAGE_COPS={'osc_decay':[np.sqrt(kappa)*aC],'osc_deph':[np.sqrt(kappa_phi)*(aC.dag()*aC)]}

opts_dmg=Options(nsteps=100000,atol=1e-8,rtol=1e-6)
def damage(rho_cav,channel,t):
    if t<=0: return rho_cav
    H0=0*num(Ncav)
    nsub=max(2,int(t/20.0)+1); tl=np.linspace(0,t,nsub)
    return mesolve(H0,rho_cav,tl,c_ops=DAMAGE_COPS[channel],options=opts_dmg).states[-1]

def readout(scheme,rho_cav,cond):
    rho=tensor(rho_cav,ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in scheme:
        if kind=='M': rho=op*rho*op.dag(); rho=rho/rho.tr()
        elif cops is None: rho=op*rho*op.dag()
        else: rho=mesolve(K/t,rho,[0,t],c_ops=cops,options=opts).states[-1]
    return rho

def calibrate_D(scheme,cond):
    # calibrate from RAW logical(0) (beta=0) -- matches run_gcrbb1_noise.py line 49 (fixed defect #1)
    rho=readout(scheme,ket2dm(L0raw),cond); ro=rho.ptrace(0)
    bx=float(np.real(expect(xO,ro)))/np.sqrt(2); bp=float(np.real(expect(pO,ro)))/np.sqrt(2)
    corr=1j*(bp+np.sqrt(np.pi/2)/2)-bx
    return displace(Ncav,corr)

# ---------------------------------------------------------------------------------------------------
ckpt_path='Paper_Data/damage_readout.npz'
assert os.path.exists(ckpt_path), f"missing {ckpt_path}"
DATA=dict(np.load(ckpt_path))
times=DATA['times']
print(f"[{time.time()-t0:.0f}s] loaded {ckpt_path}: {len(DATA)} keys, times={np.round(times,1)}",flush=True)

# snapshot original err keys for the byte-identity check at the end
ERR_ORIG={k:np.array(v,copy=True) for k,v in DATA.items() if k.endswith('|err')}
N_KEYS_ORIG=len(DATA)

# ---- cache the 28 protocol-independent damaged states once ----
print(f"[{time.time()-t0:.0f}s] caching {len(CHANNELS)*len(STATES)*len(times)} damaged states...",flush=True)
DMG={}
for channel in CHANNELS:
    for stname in STATES:
        rho0=ket2dm(STATES[stname])
        for it,t in enumerate(times):
            DMG[(channel,stname,it)]=damage(rho0,channel,t)
            print(f"[{time.time()-t0:.0f}s]   cached damage {channel:9s} {stname} t={t:7.1f}",flush=True)
print(f"[{time.time()-t0:.0f}s] damage cache complete ({len(DMG)} states)",flush=True)

# ---- recompute per protocol ----
max_err_diff=0.0
new_infid={}   # staged {key: array}; only written after full-run err validation passes
noiseless_t0={}   # {proto: {stname: infid}}
for proto in PROTOCOLS:
    scheme=build_scheme(proto)
    Dcache={cond:calibrate_D(scheme,cond) for cond in READ_CONDS}
    print(f"[{time.time()-t0:.0f}s] {proto}: scheme built ({len(scheme)} seg), D calibrated (raw logical0)",flush=True)
    noiseless_t0[proto]={}
    for stname in STATES:
        for channel in CHANNELS:
            errs={cond:np.zeros(len(times)) for cond in READ_CONDS}
            infids={cond:np.zeros(len(times)) for cond in READ_CONDS}
            for it,t in enumerate(times):
                rho_dmg=DMG[(channel,stname,it)]
                for cond in READ_CONDS:
                    rho_out=readout(scheme,rho_dmg,cond)
                    p1=float(np.real(expect(pydm,rho_out.ptrace(1))))
                    err=(1-p1) if stname=='L0' else p1
                    ro_out=rho_out.ptrace(0); D=Dcache[cond]
                    infid=1-float(fidelity(PREF[stname],D*ro_out*D.dag()))
                    errs[cond][it]=err; infids[cond][it]=infid
            for cond in READ_CONDS:
                stored=DATA[f'{proto}|{channel}|{cond}|{stname}|err']
                d=float(np.max(np.abs(errs[cond]-stored)))
                if d>max_err_diff: max_err_diff=d
                if d>2e-3:
                    print(f"[{time.time()-t0:.0f}s] ABORT: err mismatch for "
                          f"{proto}|{channel}|{cond}|{stname}: max|diff|={d:.3e} > 2e-3.\n"
                          f"   recomputed={np.round(errs[cond],5)}\n   stored    ={np.round(stored,5)}\n"
                          f"   Cached damage states or scheme build diverge from the original run; "
                          f"NOT overwriting.",flush=True)
                    sys.exit(1)
                new_infid[f'{proto}|{channel}|{cond}|{stname}|infid']=infids[cond]
            if channel=='osc_decay':
                noiseless_t0[proto][stname]=infids['noiseless'][0]
    print(f"[{time.time()-t0:.0f}s] {proto}: recomputed OK (running max err-abs-diff={max_err_diff:.3e})",flush=True)
    # checkpoint: apply this protocol's infid keys and save
    for k,v in new_infid.items():
        if k.startswith(proto+'|'): DATA[k]=v
    np.savez(ckpt_path,**DATA)
    print(f"[{time.time()-t0:.0f}s] {proto}: checkpoint saved ({len(DATA)} keys)",flush=True)

# ---- final integrity + sanity report ----
DATA=dict(np.load(ckpt_path))
err_pass=True
for k,v in ERR_ORIG.items():
    if not np.array_equal(np.asarray(DATA[k]),v): err_pass=False; print(f"   ERR CHANGED: {k}",flush=True)

print("\n================= SANITY REPORT =================",flush=True)
print(f"max err-abs-diff (recomputed vs stored): {max_err_diff:.3e}  (threshold 2e-3)",flush=True)
print(f"final key count: {len(DATA)}  (was {N_KEYS_ORIG})",flush=True)
print(f"err preserved: {'PASS' if err_pass else 'FAIL'}",flush=True)
print("\nt=0 NOISELESS post-readout infidelity (osc_decay channel, it=0):",flush=True)
print(f"  {'protocol':18s} {'L0':>10s} {'L1':>10s}",flush=True)
for proto in PROTOCOLS:
    print(f"  {proto:18s} {noiseless_t0[proto]['L0']:10.5f} {noiseless_t0[proto]['L1']:10.5f}",flush=True)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
