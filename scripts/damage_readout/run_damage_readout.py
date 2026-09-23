"""Damage-then-readout analysis: take UNDISPLACED (in the eps-sweep sense -- see canonicalization note
below) logical-0/1 GKP codewords, damage them with a pure oscillator decoherence channel for a range of
times (no readout happening yet), then read them out with each of the 7 kick/correction protocols under
4 readout-time noise conditions. Records readout error and post-readout oscillator infidelity vs damage
time, for both damage channels. INFIDELITY = 1 - fidelity(PREF, D*ro_out*D.dag()), where PREF is the
PRISTINE canonical codeword for the run's logical state (ket2dm(L0) or ket2dm(L1)) -- NOT the damaged
input -- and D is calibrated from the RAW logical(0) (beta=0) back-action (see calibrate_D). At t=0 this
reduces to the readout-only back-action infidelity (reproducing the fidelity table) and rises
monotonically with damage.

CODEWORD CANONICALIZATION NOTE: literally using logical(mu) undisplaced (beta=0) gives EXACTLY qerr=0.5
for every protocol at t=0 (verified numerically) -- the raw comb built by logical(mu) is centered at the
readout ancilla's discrimination midpoint, not its zero-error point. Throughout run_mesolve_4.py /
run_gcrbb1_noise.py the actual zero-error reference codeword is displace(-shift)*logical(mu)
(shift=sqrt(pi/2)/2; this is exactly their eps=0, beta=-shift convention). We therefore define
  L0 = (displace(Ncav,-shift)*logical(0)).unit(),  L1 = (displace(Ncav,-shift)*logical(1)).unit()
i.e. "undisplaced" = no additional eps-type coherent-displacement error on top of this canonical
zero-error codeword; only the decoherence DAMAGE channel (below) perturbs the state before readout.
This reproduces small t=0 errors (e.g. ~1e-5 for BB1(GCR)-QITE) matching the other scripts' conventions.

DAMAGE CHANNELS (applied to the bare Ncav-dim cavity density matrix, zero Hamiltonian, one collapse op,
for time t -- precedent: test_round_x_purify.py lines 24-29):
  osc_decay: sqrt(kappa)*aC        kappa=1/1000 us^-1        (Sivak-model oscillator photon loss)
  osc_deph : sqrt(kappa_phi)*(aC^dag aC)   kappa_phi=1/5000 us^-1  (oscillator dephasing rate; NONE in
             the Sivak model used elsewhere in this repo -- chosen here as a comparison rate.)

DAMAGE TIME GRID: shared np.linspace(0,1000.0,7) us for BOTH channels (comparable x-axis). Checked
numerically (Ncav=100, GCR/QITE, noiseless readout): osc_decay readout error rises to ~0.46-0.47 by
t=1000us (visibly degraded, still short of its ~0.5 asymptote -- NOT fully saturated), while osc_deph
is at ~0.22-0.23 (clearly slower, as expected for the smaller rate). 1000us keeps decay informative
without full saturation, so the single shared grid is used unmodified.

READOUT-TIME noise conditions (4, NOTE: no 'complete'): noiseless, tr_deph, tr_decay, osc_decay -- same
COPS entries as run_mesolve_4.py.

Output: Paper_Data/damage_readout.npz (NEW file, never merged into mesolve_4.npz), keys
  '{protocol}|{damage_channel}|{readout_cond}|{state}|err'    (array over the 7 damage times)
  '{protocol}|{damage_channel}|{readout_cond}|{state}|infid'  (array over the 7 damage times)
  plus 'times', 'kappa', 'kappa_phi'.
protocol in {GCR, BB1(GCR)-QITE, BB1, GCR-BB1, GCR-BB1-blockA, GCR-BB1-blockB, herald-RUS}
damage_channel in {osc_decay, osc_deph}; readout_cond in the 4-list above; state in {L0, L1}.

CHECKPOINTING: after each protocol (outer loop) finishes, the accumulated dict is saved to
Paper_Data/damage_readout.npz. On a non---test startup, an existing file is loaded and any protocol
whose full key-set is already present is skipped (resume after a kill).

--test: writes /tmp/damage_readout_test.npz with a 3-point time grid; never touches Paper_Data.
CLI: argv[1]=Ncav (default 400). argv[2] (if given) is unused (kept for CLI-shape symmetry)."""
import os, sys, time, numpy as np
from qutip import *
t0=time.time(); TEST='--test' in sys.argv
_pos=[a for a in sys.argv[1:] if a!='--test']
Ncav=int(_pos[0]) if len(_pos)>0 else 400
gam=1/200.; gamphi=1/200.; kappa=1/1000.
kappa_phi=1/5000.   # oscillator dephasing rate; none in Sivak model, chosen 1/5000 us^-1
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
L0raw=logical(0)   # RAW comb (beta=0); used to calibrate D (matches run_gcrbb1_noise.py calibrate_D)
STATES={'L0':L0,'L1':L1}
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} TEST={TEST} shift={shift:.5f} kappa={kappa} kappa_phi={kappa_phi}",flush=True)

# ---- BB1 / kick / det_corr / U / Mexact -- copied verbatim from run_mesolve_4.py ----
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

opts_dmg=Options(nsteps=100000,atol=1e-8,rtol=1e-6)   # stiffer: a^dag a dephasing over long idle needs many substeps
def damage(rho_cav,channel,t):
    if t<=0: return rho_cav
    H0=0*num(Ncav)
    # subdivide [0,t] (~every 20us) so each sub-interval gets its own nsteps budget (avoids ODE substep overflow)
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
    # fixed per (protocol,readout_cond) calibration displacement from the RAW logical(0) (beta=0,
    # un-shifted comb -- L0raw), exactly as calibrate_D() in run_gcrbb1_noise.py line 49 (which uses
    # run(0.0,...) => displace(0)*logical(0)). The corr formula below already bakes in the +shift
    # (via +sqrt(pi/2)/2), so calibrating from L0=displace(-shift)*logical(0) would apply the shift
    # twice and mis-calibrate D; the raw logical(0) is the correct convention and reproduces the
    # note's fidelity-table values. Generalized from cond='noiseless' to the actual readout_cond being
    # evaluated, since with cond!='noiseless' the readout mesolve itself carries decoherence even at
    # zero damage time -- this D is reused across all damage times.
    rho=readout(scheme,ket2dm(L0raw),cond); ro=rho.ptrace(0)
    bx=float(np.real(expect(xO,ro)))/np.sqrt(2); bp=float(np.real(expect(pO,ro)))/np.sqrt(2)
    corr=1j*(bp+np.sqrt(np.pi/2)/2)-bx
    return displace(Ncav,corr)

if TEST:
    times=np.linspace(0,1000.0,3)
    ckpt_path='/tmp/damage_readout_test.npz'
    DATA={}
else:
    times=np.linspace(0,1000.0,7)
    ckpt_path='Paper_Data/damage_readout.npz'
    if os.path.exists(ckpt_path):
        DATA=dict(np.load(ckpt_path))
        print(f"[{time.time()-t0:.0f}s] resumed from {ckpt_path} ({len(DATA)} existing keys)",flush=True)
    else:
        DATA={}
print(f"[{time.time()-t0:.0f}s] damage time grid (us): {np.round(times,2)}",flush=True)

def protocol_done(name):
    for ch in CHANNELS:
        for cond in READ_CONDS:
            for st in STATES:
                if f'{name}|{ch}|{cond}|{st}|err' not in DATA: return False
                if f'{name}|{ch}|{cond}|{st}|infid' not in DATA: return False
    return True

for proto in PROTOCOLS:
    if not TEST and protocol_done(proto):
        print(f"[{time.time()-t0:.0f}s] {proto}: already complete in checkpoint, skipping",flush=True)
        continue
    scheme=build_scheme(proto)
    print(f"[{time.time()-t0:.0f}s] {proto}: scheme built ({len(scheme)} segments)",flush=True)
    Dcache={cond:calibrate_D(scheme,cond) for cond in READ_CONDS}
    print(f"[{time.time()-t0:.0f}s] {proto}: calibration D computed for {READ_CONDS}",flush=True)
    PREF={'L0':ket2dm(L0),'L1':ket2dm(L1)}   # pristine canonical codewords (infidelity reference)
    for stname,Lst in STATES.items():
        rho0=ket2dm(Lst)
        for channel in CHANNELS:
            errs={cond:np.zeros(len(times)) for cond in READ_CONDS}
            infids={cond:np.zeros(len(times)) for cond in READ_CONDS}
            for it,t in enumerate(times):
                rho_dmg=damage(rho0,channel,t)
                for cond in READ_CONDS:
                    rho_out=readout(scheme,rho_dmg,cond)
                    p1=float(np.real(expect(pydm,rho_out.ptrace(1))))
                    err=(1-p1) if stname=='L0' else p1
                    ro_out=rho_out.ptrace(0); D=Dcache[cond]
                    infid=1-float(fidelity(PREF[stname],D*ro_out*D.dag()))
                    errs[cond][it]=err; infids[cond][it]=infid
                print(f"[{time.time()-t0:.0f}s] {proto:16s} {channel:10s} {stname} t={t:7.1f} "
                      f"err[{','.join(cond+':'+format(errs[cond][it],'.3e') for cond in READ_CONDS)}]",flush=True)
            for cond in READ_CONDS:
                DATA[f'{proto}|{channel}|{cond}|{stname}|err']=errs[cond]
                DATA[f'{proto}|{channel}|{cond}|{stname}|infid']=infids[cond]
    DATA['times']=times; DATA['kappa']=np.array(kappa); DATA['kappa_phi']=np.array(kappa_phi)
    np.savez(ckpt_path,**DATA)
    print(f"[{time.time()-t0:.0f}s] {proto}: DONE, checkpoint saved to {ckpt_path} ({len(DATA)} keys)",flush=True)

if TEST:
    print(f"[{time.time()-t0:.0f}s] TEST mode summary ({len(DATA)} keys):",flush=True)
    for k,v in DATA.items():
        arr=np.asarray(v)
        print(f"   {k:45s} shape={arr.shape} finite={np.all(np.isfinite(arr))}",flush=True)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
