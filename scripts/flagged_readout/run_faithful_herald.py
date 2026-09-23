"""Faithful heralded readout: true post-selection (measure+discard) for the 3 non-unitary "RUS"
corrections of the herald-RUS scheme, replacing the old exact-M+renormalize (which is NOT true
post-selection -- it never discards errored shots and its renormalization amplifies decoherence).

Circuit per correction i=0,1,2: herald-ancilla starts in |g>. Apply the block-encoding gate
W_i = exp(-i * c_i * pO(x)SIG(ph_i)(x)sigma_y_anc) under NOISE (mesolve, including decoherence on
the ancilla at the same transmon rates as the readout qubit), then a (noiseless, virtual) Hadamard
on the ancilla, then PROJECT the ancilla onto |g> (success), renormalize, and multiply the running
success probability by the projected trace. The 4th step is the deterministic GCR target (no herald).
Kicks are mesolve'd with noise as usual. Hilbert space: oscillator(x)readout-qubit(x)herald-ancilla,
Ncav x 2 x 2.
"""
import sys, time, numpy as np
from qutip import *

t0=time.time()
Ncav = int(sys.argv[1]) if len(sys.argv)>1 else 100
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav)
xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)

def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); shift=np.sqrt(np.pi/2)/2; u=np.sqrt(np.pi)/(2*np.sqrt(2))
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]

# ---- 3-body (osc x readout-qubit x ancilla) embeddings ----
I2=qeye(2)
def emb2(op2body):  # op2body acts on osc x readout-qubit -> embed x I(ancilla)
    return tensor(op2body, I2)
def kick(i):
    th,ph=BB1[i]
    K2 = (th/sq)*tensor(xO,SIG(ph)); t=th/np.sqrt(2*np.pi)
    return emb2(K2), t
def det_corr(c,nu):
    K2 = c*tensor(pO,SIG(nu)); t=abs(c)/np.sqrt(2)
    return emb2(K2), t
def herald_W(i):
    th,ph=BB1[i]; c=th*ep/sq
    G = c*tensor(pO, SIG(ph), sigmay())    # 3-body generator, Hermitian -> exp(-iG) unitary
    t = abs(c)/np.sqrt(2)
    return G, t

def U3(K): return (-1j*K).expm(method='dense')
Had_anc = tensor(qeye(Ncav), I2, Qobj([[1,1],[1,-1]])/np.sqrt(2))
Pg_anc  = tensor(qeye(Ncav), I2, ket2dm(basis(2,0)))

sm_r=tensor(qeye(Ncav),sigmam(),I2); sz_r=tensor(qeye(Ncav),sigmaz(),I2)
sm_a=tensor(qeye(Ncav),I2,sigmam()); sz_a=tensor(qeye(Ncav),I2,sigmaz())
aa3 =tensor(aC,I2,I2)
COPS3={'noiseless':None,
       'complete':[np.sqrt(gam)*sm_r,np.sqrt(gamphi/2)*sz_r,np.sqrt(gam)*sm_a,np.sqrt(gamphi/2)*sz_a,np.sqrt(kappa)*aa3],
       'tr_decay':[np.sqrt(gam)*sm_r,np.sqrt(gam)*sm_a],
       'tr_deph':[np.sqrt(gamphi/2)*sz_r,np.sqrt(gamphi/2)*sz_a],
       'osc_decay':[np.sqrt(kappa)*aa3]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)

def evolve_gate(rho, K, t, cops):
    if cops is None:
        return U3(K)*rho*U3(K).dag()
    return mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]

def faithful_herald_readout(beta, cond):
    psi=(displace(Ncav,beta)*L0).unit()
    rho = tensor(ket2dm(psi), ket2dm(gq), ket2dm(gq))   # osc x readout-qubit(g) x ancilla(g)
    cops = COPS3[cond]
    p_cum = 1.0
    for i in range(4):
        if i < 3:
            G, t = herald_W(i)
            rho = evolve_gate(rho, G, t, cops)
            rho = Had_anc*rho*Had_anc.dag()
            rho_g = Pg_anc*rho*Pg_anc
            p_i = float(np.real(rho_g.tr()))
            rho = rho_g/p_i
            p_cum *= p_i
        else:
            Kd, td = det_corr((np.pi/2)*ep/sq, 0+np.pi/2)
            rho = evolve_gate(rho, Kd, td, cops)
        Kk, tk = kick(i)
        rho = evolve_gate(rho, Kk, tk, cops)
    # readout qubit is index 1; ancilla (index 2) is in |g> post-projection, trace it out
    rho2 = rho.ptrace([0,1])
    qerr = 1-float(np.real(expect(pydm, rho2.ptrace(1))))
    return qerr, p_cum, rho2.ptrace(0)

conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav}  faithful heralded readout, eps=0", flush=True)
for cond in conds:
    qerr, psucc, ro = faithful_herald_readout(-shift, cond)
    print(f"[{time.time()-t0:.0f}s] cond={cond:10s} accepted_qerr={qerr:.4e}  p_success(cum)={psucc:.4e}", flush=True)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
