"""Add the two MISSING readout-time noise conditions to Paper_Data/damage_readout.npz:
      'complete' (all channels: transmon decay + transmon dephasing + oscillator decay) and
      'osc_deph' (oscillator dephasing during the readout circuit, sqrt(kappa_phi)*n).
The npz already holds noiseless / tr_decay / tr_deph / osc_decay. This script computes err+infid
for the two new conditions across 7 protocols x 2 damage channels x 2 states x 7 times and MERGES
them in without touching any existing key. As a scheme-build sanity check it ALSO recomputes the
NOISELESS condition and aborts if it disagrees with the stored noiseless err (atol=2e-3) -- so a
divergent scheme build can never silently corrupt the new conditions. Definitions match
recompute_damage_infid.py exactly: D calibrated from RAW logical(0); infidelity vs PRISTINE PREF.
Ncav=400, checkpoint after each protocol."""
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
L0raw=logical(0)
STATES={'L0':L0,'L1':L1}
PREF={'L0':ket2dm(L0),'L1':ket2dm(L1)}
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} shift={shift:.5f} kappa={kappa} kappa_phi={kappa_phi}",flush=True)

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
nn=aa.dag()*aa   # cavity number operator on full space (for oscillator dephasing during readout)
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'osc_deph':[np.sqrt(kappa_phi)*nn]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)
NEW_CONDS=['complete','osc_deph']          # the two conditions to ADD
VALIDATE_COND='noiseless'                   # recomputed only to sanity-check scheme build
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
    rho=readout(scheme,ket2dm(L0raw),cond); ro=rho.ptrace(0)
    bx=float(np.real(expect(xO,ro)))/np.sqrt(2); bp=float(np.real(expect(pO,ro)))/np.sqrt(2)
    corr=1j*(bp+np.sqrt(np.pi/2)/2)-bx
    return displace(Ncav,corr)

ckpt_path='Paper_Data/damage_readout.npz'
assert os.path.exists(ckpt_path), f"missing {ckpt_path}"
DATA=dict(np.load(ckpt_path))
times=DATA['times']
N_KEYS_ORIG=len(DATA)
print(f"[{time.time()-t0:.0f}s] loaded {ckpt_path}: {N_KEYS_ORIG} keys, times={np.round(times,1)}",flush=True)
KEYS_ORIG=set(DATA.keys())

# cache the 28 protocol-independent damaged states once
print(f"[{time.time()-t0:.0f}s] caching {len(CHANNELS)*len(STATES)*len(times)} damaged states...",flush=True)
DMG={}
for channel in CHANNELS:
    for stname in STATES:
        rho0=ket2dm(STATES[stname])
        for it,t in enumerate(times):
            DMG[(channel,stname,it)]=damage(rho0,channel,t)
    print(f"[{time.time()-t0:.0f}s]   cached damage {channel}",flush=True)
print(f"[{time.time()-t0:.0f}s] damage cache complete ({len(DMG)} states)",flush=True)

max_val_diff=0.0
CONDS_TO_RUN=NEW_CONDS+[VALIDATE_COND]
for proto in PROTOCOLS:
    scheme=build_scheme(proto)
    Dcache={cond:calibrate_D(scheme,cond) for cond in CONDS_TO_RUN}
    print(f"[{time.time()-t0:.0f}s] {proto}: scheme built ({len(scheme)} seg), D calibrated",flush=True)
    staged={}
    for stname in STATES:
        for channel in CHANNELS:
            for cond in CONDS_TO_RUN:
                errs=np.zeros(len(times)); infids=np.zeros(len(times))
                for it,t in enumerate(times):
                    rho_out=readout(scheme,DMG[(channel,stname,it)],cond)
                    p1=float(np.real(expect(pydm,rho_out.ptrace(1))))
                    err=(1-p1) if stname=='L0' else p1
                    ro_out=rho_out.ptrace(0); D=Dcache[cond]
                    infid=1-float(fidelity(PREF[stname],D*ro_out*D.dag()))
                    errs[it]=err; infids[it]=infid
                if cond==VALIDATE_COND:
                    stored=DATA[f'{proto}|{channel}|{VALIDATE_COND}|{stname}|err']
                    dd=float(np.max(np.abs(errs-stored)))
                    if dd>max_val_diff: max_val_diff=dd
                    if dd>2e-3:
                        print(f"[{time.time()-t0:.0f}s] ABORT: noiseless build check failed for "
                              f"{proto}|{channel}|{stname}: max|diff|={dd:.3e} > 2e-3. NOT writing.",flush=True)
                        sys.exit(1)
                else:
                    staged[f'{proto}|{channel}|{cond}|{stname}|err']=errs
                    staged[f'{proto}|{channel}|{cond}|{stname}|infid']=infids
    # guard: never overwrite an existing key
    clash=[k for k in staged if k in KEYS_ORIG]
    assert not clash, f"would clobber existing keys: {clash[:3]}..."
    DATA.update(staged)
    np.savez(ckpt_path,**DATA)
    print(f"[{time.time()-t0:.0f}s] {proto}: OK (noiseless build max-diff={max_val_diff:.3e}); "
          f"checkpoint saved ({len(DATA)} keys, +{len(DATA)-N_KEYS_ORIG})",flush=True)

DATA=dict(np.load(ckpt_path))
print("\n================= REPORT =================",flush=True)
print(f"noiseless build max-abs-diff vs stored: {max_val_diff:.3e}  (threshold 2e-3)",flush=True)
print(f"final key count: {len(DATA)}  (was {N_KEYS_ORIG}, +{len(DATA)-N_KEYS_ORIG})",flush=True)
new_conds=sorted(set(k.split('|')[2] for k in DATA if '|' in k and k.count('|')>=4))
print(f"readout conditions now present: {new_conds}",flush=True)
print("\nt=0 P(e) for the two NEW conditions (osc_decay damage channel):",flush=True)
for cond in NEW_CONDS:
    print(f"  --- {cond} ---",flush=True)
    for proto in PROTOCOLS:
        e0=DATA[f'{proto}|osc_decay|{cond}|L0|err']; e1=DATA[f'{proto}|osc_decay|{cond}|L1|err']
        avg=(e0+e1)/2
        print(f"    {proto:16s} t=0: {avg[0]:.4e}   t=max: {avg[-1]:.4e}",flush=True)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
