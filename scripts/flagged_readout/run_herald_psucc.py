"""Supplementary computation: herald-RUS success probability.
run_mesolve_4.py applies the herald-RUS scheme's 3 non-unitary M corrections as instantaneous
exact projectors, renormalizing (rho=rho/rho.tr()) at each step -- rho.tr() just before
renormalizing IS the herald success probability of that sub-step (probability the RUS attempt
would not have been rejected). This was not stored in mesolve_4.npz. This script recomputes it
(same SCHEMES/COPS/eps-grid/beta convention as run_mesolve_4.py) and records the cumulative
success probability (product over the 3 M steps) for the herald-RUS case, for both panels and
all 5 noise conditions, at Ncav=400, NE=8 (matching mesolve_4.npz's grid).
"""
import sys, time, numpy as np
from qutip import *
t0=time.time(); Ncav=int(sys.argv[1]) if len(sys.argv)>1 else 400
NE=int(sys.argv[2]) if len(sys.argv)>2 else 8
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0)
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); u=np.sqrt(np.pi)/(2*np.sqrt(2)); shift=np.sqrt(np.pi/2)/2
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph=BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph=BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def U(K): return (-1j*K).expm(method='dense')
def sc_herald():
    L=[]
    for i in range(4):
        Kk,tk=kick(i)
        if i<3: L.append(('M',Mexact(i),None,0.0))
        else:   K,t=det_corr((np.pi/2)*ep/sq,0+np.pi/2); L.append(('U',U(K),K,t))
        L.append(('U',U(Kk),Kk,tk))
    return L
HERALD=sc_herald()
sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)
def psucc(beta, cond):
    psi=(displace(Ncav,beta)*L0).unit(); pdm=ket2dm(psi)
    rho=tensor(pdm,ket2dm(gq)); cops=COPS[cond]
    p_cum=1.0
    for kind,op,K,t in HERALD:
        if kind=='M':
            rho=op*rho*op.dag(); p_step=float(np.real(rho.tr())); p_cum*=p_step
            rho=rho/rho.tr()
        elif cops is None: rho=op*rho*op.dag()
        else: rho=mesolve(K/t,rho,[0,t],c_ops=cops,options=opts).states[-1]
    return p_cum
epsA=np.linspace(0,u,NE); epsB=np.linspace(2*u,3*u,NE)
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
DATA={'epsA':epsA,'epsB':epsB,'u':u}
for cond in conds:
    pA=np.array([psucc(e-shift,cond) for e in epsA])
    pB=np.array([psucc(e-shift,cond) for e in epsB])
    DATA[f'herald-RUS|{cond}|a|psucc']=pA
    DATA[f'herald-RUS|{cond}|b|psucc']=pB
    print(f"[{time.time()-t0:.0f}s] {cond:10s} psucc_a_mean={pA.mean():.4f} psucc_b_mean={pB.mean():.4f}",flush=True)
    np.savez('Paper_Data/herald_psucc.npz',**DATA)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
