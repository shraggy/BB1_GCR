"""TASK A: For each of the 4 existing readout cases (GCR, BB1(GCR)-QITE, BB1, herald-RUS) and each
NOISY condition (complete, tr_decay, tr_deph, osc_decay -- noiseless skipped, impurity ~0), take the
post-readout oscillator state at eps=0, apply up to 10 NOISY alternating round_x/round_p sBs rounds
(order='xp', full sequence, no early stop), track impurity, and report the ARGMIN (lowest-impurity
round, not necessarily round 10, since the alternating trace can be non-monotonic). Ncav=400.
Readout error is unchanged from mesolve_4.npz (sBs correction only affects the oscillator state;
Task A does not combine repeated reads, so the round_p logical-flip Pauli-frame issue does not
apply here -- that tracking is only needed in Task B/C).
"""
import sys, time, numpy as np
from qutip import *
import sbs_lib as sb

t0=time.time()
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 400
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav}", flush=True)

gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
L0 = sb.logical(0, Ncav)
shift = np.sqrt(np.pi/2)/2
qp = np.load('Paper_Data/qite_det_params.npy')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph=BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph=BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def U(K): return (-1j*K).expm(method='dense')
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
sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)

def postreadout_rho(scheme, beta, cond):
    psi=(displace(Ncav,beta)*L0).unit()
    rho=tensor(ket2dm(psi),ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in scheme:
        if kind=='M': rho=op*rho*op.dag(); rho=rho/rho.tr()
        elif cops is None: rho=op*rho*op.dag()
        else: rho=mesolve(K/t,rho,[0,t],c_ops=cops,options=opts).states[-1]
    return rho.ptrace(0)

conds=['complete','tr_decay','tr_deph','osc_decay']
DATA={}
for scn,scheme in SCHEMES.items():
    for cond in conds:
        ro = postreadout_rho(scheme, -shift, cond)   # eps=0 -> beta = eps-shift = -shift
        im0 = sb.impurity(ro)
        trace, rho_opt, minimp, argmin_round, n_p_used = sb.run_sbs_sequence(ro, cond, Ncav, nrounds=10, order='xp', early_stop=False)
        DATA[f'{scn}|{cond}|impur0']=im0
        DATA[f'{scn}|{cond}|trace']=np.array(trace)
        DATA[f'{scn}|{cond}|minimp']=minimp
        DATA[f'{scn}|{cond}|argmin_round']=argmin_round
        reduction = im0 - minimp
        print(f"[{time.time()-t0:.0f}s] {scn:14s} {cond:10s} impur_before={im0:.4e} minimp_after_sBs={minimp:.4e} reduction={reduction:.4e} argmin_round={argmin_round} nrounds_run={len(trace)-1}",flush=True)
        np.savez('Paper_Data/task_A_sbs.npz',**DATA)
print(f"[{time.time()-t0:.0f}s] TASK A DONE",flush=True)
