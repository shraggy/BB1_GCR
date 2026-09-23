"""TASK B: new 5th case "GCR+sBs (repeated)".
Circuit: GCR readout -> N_sbs NOISY round_x-only sBs rounds (full sequence, no early stop) -> GCR
readout -> ... for NBLOCKS GCR reads total, interleaved with NBLOCKS-1 sBs correction blocks.
The repeated GCR reads (each giving an estimate P_i=P(+1) of the same logical value) are combined
via the log-odds (soft, Bayesian-optimal for independent estimates) rule:
    odds = prod_i P_i/(1-P_i);  P_combined = odds/(1+odds)
Records combined readout error (1-P_combined) and final oscillator impurity, across the Fig-3 eps
ranges (reusing epsA/epsB from mesolve_4.npz), for all 5 noise conditions. Ncav=400.
"""
import sys, time, numpy as np
from qutip import *
import sbs_lib as sb

t0=time.time()
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 400
NE   = int(sys.argv[2]) if len(sys.argv)>2 else 5
NBLOCKS = int(sys.argv[3]) if len(sys.argv)>3 else 2   # number of GCR reads (blocks-1 sBs corrections between)
NSBS = int(sys.argv[4]) if len(sys.argv)>4 else 6       # cap on sBs rounds per correction block
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} NE={NE} NBLOCKS={NBLOCKS} NSBS={NSBS}",flush=True)

gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; sq=np.sqrt(np.pi)
L0 = sb.logical(0, Ncav)
u=np.sqrt(np.pi)/(2*np.sqrt(2)); shift=np.sqrt(np.pi/2)/2

def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def kick3(): th=np.pi/2; return (th/sq)*tensor(xO,SIG(0.0)), th/np.sqrt(2*np.pi)
def U(K): return (-1j*K).expm(method='dense')
def sc_gcr():
    K,t=det_corr((np.pi/2)*ep/sq, 0+np.pi/2); Kk,tk=kick3()
    return [('U',U(K),K,t),('U',U(Kk),Kk,tk)]
GCR = sc_gcr()
sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)

def apply_gcr_readout(rho_osc, cond):
    """Apply GCR scheme to oscillator (fresh |g> ancilla), return (P_plus1, rho_osc_after)."""
    rho = tensor(rho_osc, ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in GCR:
        if cops is None: rho = op*rho*op.dag()
        else: rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    P1 = float(np.real(expect(pydm, rho.ptrace(1))))
    return P1, rho.ptrace(0)

def combine_logodds(Ps):
    Ps = np.clip(np.array(Ps), 1e-9, 1-1e-9)
    odds = np.prod(Ps/(1-Ps))
    return odds/(1+odds)

def run_case(beta, cond):
    """Repeated GCR reads interleaved with alternating round_x/round_p sBs blocks. round_p carries
    a deterministic logical-X back-action (flips logical(0)<->logical(1) every application), so we
    track the accumulated Pauli-frame parity (count of round_p rounds actually applied, i.e. up to
    the argmin round of each block, mod 2) and reinterpret each subsequent GCR read's P(+1) as
    1-P(+1) if the frame is flipped relative to the FIRST read, before combining via log-odds."""
    psi = (displace(Ncav, beta)*L0).unit()
    ro = ket2dm(psi)
    Ps = []
    flip_parity = 0
    argmin_rounds = []
    impur_final = 0.0
    for b in range(NBLOCKS):
        P, ro = apply_gcr_readout(ro, cond)
        P_eff = (1.0-P) if (flip_parity % 2) else P
        Ps.append(P_eff)
        if b < NBLOCKS-1:
            trace, ro, minimp, argmin_round, n_p_used = sb.run_sbs_sequence(ro, cond, Ncav, nrounds=NSBS, order='xp', early_stop=False)
            flip_parity += n_p_used
            argmin_rounds.append(argmin_round)
        impur_final = sb.impurity(ro)
    Pc = combine_logodds(Ps)
    return 1-Pc, impur_final, Ps, argmin_rounds

epsA = np.linspace(0,u,NE); epsB = np.linspace(2*u,3*u,NE)
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
DATA={'epsA':epsA,'epsB':epsB,'u':u,'NBLOCKS':NBLOCKS,'NSBS':NSBS}
for cond in conds:
    rA=[]; rB=[]
    for e in epsA:
        q,im,Ps,argmins = run_case(e-shift, cond); rA.append((q,im))
        print(f"[{time.time()-t0:.0f}s] a cond={cond:10s} eps={e:.4f} q={q:.4e} impur={im:.4e} Ps={np.round(Ps,4)} argmin_rounds={argmins}",flush=True)
    for e in epsB:
        q,im,Ps,argmins = run_case(e-shift, cond); rB.append((1-q,im))  # store as readout error convention consistent w/ mesolve_4 (b panel uses 1-q)
        print(f"[{time.time()-t0:.0f}s] b cond={cond:10s} eps={e:.4f} q={1-q:.4e} impur={im:.4e} Ps={np.round(Ps,4)} argmin_rounds={argmins}",flush=True)
    rA=np.array(rA); rB=np.array(rB)
    DATA[f'GCR+sBs|{cond}|a|q']=rA[:,0]; DATA[f'GCR+sBs|{cond}|a|impur']=rA[:,1]
    DATA[f'GCR+sBs|{cond}|b|q']=rB[:,0]; DATA[f'GCR+sBs|{cond}|b|impur']=rB[:,1]
    np.savez('Paper_Data/task_B_gcrsbs.npz',**DATA)
print(f"[{time.time()-t0:.0f}s] TASK B DONE",flush=True)
