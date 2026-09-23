"""TASK C: heralded variant of Task B ("GCR+sBs (repeated), heralded" -- calibration case).
Same repeated-GCR-read + interleaved-sBs structure as Task B (round_x ONLY, round_p dropped),
EXCEPT the interleaved sBs rounds are heralded: after each sBs round, the ancilla is post-selected
on |g> (projected + renormalized) before being reset for the next round; the cumulative herald
success probability is tracked (product of each round's conditional success probability), full
sequence (no early stop). The two GCR reads themselves are combined via the same log-odds soft-
combination as Task B (not heralded -- GCR is intended to be near-deterministic/heraldable via its
own ancilla outcome already covered by herald-RUS). Compared against the existing herald-RUS case
for calibration. Ncav=400.
"""
import sys, time, numpy as np
from qutip import *
import sbs_lib as sb

t0=time.time()
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 400
NE   = int(sys.argv[2]) if len(sys.argv)>2 else 5
NBLOCKS = int(sys.argv[3]) if len(sys.argv)>3 else 2
NSBS = int(sys.argv[4]) if len(sys.argv)>4 else 6
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
    rho = tensor(rho_osc, ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in GCR:
        if cops is None: rho = op*rho*op.dag()
        else: rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    P1 = float(np.real(expect(pydm, rho.ptrace(1))))
    return P1, rho.ptrace(0)

def heralded_sbs_block(rho_osc, cond, nrounds, order='xp'):
    """Apply up to nrounds sBs rounds (alternating round_x/round_p per `order`, full sequence, no
    early stop), each heralded on ancilla=g, tracking cumulative success probability and impurity.
    Since the alternating trace can be non-monotonic, use the ARGMIN round (not necessarily the
    last) as the block's output state -- consistent with Task A/B -- and report both the number of
    rounds used and how many of those were round_p (n_p_used), which callers need to track the
    logical-X Pauli-frame parity round_p's back-action induces on subsequent GCR reads. The
    cumulative herald success probability returned is the product over only the rounds actually
    used (up to argmin_round), since rounds beyond that point are discarded."""
    trace=[sb.impurity(rho_osc)]
    states=[rho_osc]
    p_cum_list=[1.0]
    rho = rho_osc
    p_run = 1.0
    for n in range(nrounds):
        axis = order[n % len(order)]
        rho, p = sb.sbs_round(rho, axis, cond, Ncav, herald=True)
        p_run *= p
        trace.append(sb.impurity(rho))
        states.append(rho)
        p_cum_list.append(p_run)
    argmin_round = int(np.argmin(trace))
    n_p_used = sum(1 for n in range(argmin_round) if order[n % len(order)] == 'p')
    return states[argmin_round], trace, p_cum_list[argmin_round], argmin_round, n_p_used

def combine_logodds(Ps):
    Ps = np.clip(np.array(Ps), 1e-9, 1-1e-9)
    odds = np.prod(Ps/(1-Ps))
    return odds/(1+odds)

def run_case(beta, cond):
    """Same Pauli-frame-parity tracking as Task B (see run_task_B.py): round_p's deterministic
    logical-X back-action means each subsequent GCR read must be reinterpreted P<->1-P if an odd
    number of round_p rounds were actually applied (up to the argmin round) in the heralded sBs
    block(s) preceding it, before combining via log-odds."""
    psi = (displace(Ncav, beta)*L0).unit()
    ro = ket2dm(psi)
    Ps=[]; p_herald_cum = 1.0; impur_final=0.0
    flip_parity = 0
    argmin_rounds=[]
    for b in range(NBLOCKS):
        P, ro = apply_gcr_readout(ro, cond)
        P_eff = (1.0-P) if (flip_parity % 2) else P
        Ps.append(P_eff)
        if b < NBLOCKS-1:
            ro, trace, p_cum, argmin_round, n_p_used = heralded_sbs_block(ro, cond, NSBS)
            p_herald_cum *= p_cum
            flip_parity += n_p_used
            argmin_rounds.append(argmin_round)
        impur_final = sb.impurity(ro)
    Pc = combine_logodds(Ps)
    return 1-Pc, impur_final, p_herald_cum, Ps, argmin_rounds

epsA = np.linspace(0,u,NE); epsB = np.linspace(2*u,3*u,NE)
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
DATA={'epsA':epsA,'epsB':epsB,'u':u,'NBLOCKS':NBLOCKS,'NSBS':NSBS}
for cond in conds:
    rA=[]; rB=[]
    for e in epsA:
        q,im,ph,Ps,argmins = run_case(e-shift, cond); rA.append((q,im,ph))
        print(f"[{time.time()-t0:.0f}s] a cond={cond:10s} eps={e:.4f} q={q:.4e} impur={im:.4e} herald_p={ph:.4f} Ps={np.round(Ps,4)} argmin_rounds={argmins}",flush=True)
    for e in epsB:
        q,im,ph,Ps,argmins = run_case(e-shift, cond); rB.append((1-q,im,ph))
        print(f"[{time.time()-t0:.0f}s] b cond={cond:10s} eps={e:.4f} q={1-q:.4e} impur={im:.4e} herald_p={ph:.4f} Ps={np.round(Ps,4)} argmin_rounds={argmins}",flush=True)
    rA=np.array(rA); rB=np.array(rB)
    DATA[f'GCR+sBs-herald|{cond}|a|q']=rA[:,0]; DATA[f'GCR+sBs-herald|{cond}|a|impur']=rA[:,1]; DATA[f'GCR+sBs-herald|{cond}|a|psucc']=rA[:,2]
    DATA[f'GCR+sBs-herald|{cond}|b|q']=rB[:,0]; DATA[f'GCR+sBs-herald|{cond}|b|impur']=rB[:,1]; DATA[f'GCR+sBs-herald|{cond}|b|psucc']=rB[:,2]
    np.savez('Paper_Data/task_C_gcrsbs_herald.npz',**DATA)
print(f"[{time.time()-t0:.0f}s] TASK C DONE",flush=True)
