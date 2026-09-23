"""
Take the EXACT BB1(GCR) pulses and run each INDIVIDUALLY (alone, from |g>), with the CORRECT
axis, measuring fidelity and success probability vs increasing angle.

Single GCR pulse about phi (qubit starts |g> = +z, so correct precorrection axis
sigma_gamma = z_hat x phi_hat, magnitude 1 -- perpendicular). By rotational symmetry the
single-pulse performance depends only on the angle, so we sweep theta (phi=0) and mark BB1's
constituent angles {pi/2, pi, 2pi}.

Metrics (as in Fig. 2a / notebook cell 697, tensor(osc,qubit), input squeezed at Delta=0.34,
<x>=alpha=sqrt(pi)/2):
  success probability  P_succ = <(I (x) |R(theta)g><R(theta)g|)>
  success fidelity      F_H   = |<state_after_success | osc (x) R(theta)g>|^2
for bare / physical(correct axis) / ideal(vec*vecf).
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
g=basis(2,0)
Ncav=140
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); alpha=sq/2
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
init=(displace(Ncav,alpha/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()

phi=0.0
zhat=np.array([0,0,1.0]); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
gamma=svec(np.cross(zhat,phiv))  # correct physical axis = z x phi = sigma_y

def pulse(th,mode):
    ox=Opx(th,phi)
    if mode=='bare': return ox
    if mode=='phys': return ox*Opy(th,gamma)
    if mode=='ideal': return ox*Opy(th,1j*sigma(phi))

def metrics(th,mode):
    qub=((-1j*th/2*sigma(phi)).expm()*g).unit()      # target R(theta)|g>
    state=(pulse(th,mode)*tensor(init,g)).unit()
    proj=tensor(qeye(Ncav),ket2dm(qub))
    Psucc=float(np.real(expect(proj,state)))
    state3=(proj*state).unit()
    state2=tensor(init,qub)
    Fsucc=float(np.abs(state3.overlap(state2))**2)
    return Psucc,Fsucc

thetas=np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0])*np.pi
res={m:{'P':[],'F':[]} for m in ['bare','phys','ideal']}
for th in thetas:
    for m in ['bare','phys','ideal']:
        P,F=metrics(th,m); res[m]['P'].append(P); res[m]['F'].append(F)
for m in res:
    res[m]['P']=np.array(res[m]['P']); res[m]['F']=np.array(res[m]['F'])

print("Individual GCR pulse (correct axis), input Delta=0.34 at <x>=alpha:")
print("theta/pi :", " ".join(f"{t/np.pi:>6.2f}" for t in thetas))
for m in ['bare','phys','ideal']:
    print(f"\n {m}:")
    print("  1-P_succ :", " ".join(f"{1-p:>6.0e}" if (1-p)<0.001 else f"{1-p:>6.3f}" for p in res[m]['P']))
    print("  1-F_H    :", " ".join(f"{1-f:>6.0e}" if (1-f)<0.001 else f"{1-f:>6.3f}" for f in res[m]['F']))
print("\nBB1 constituent angles: pi/2 (target), pi (x2), 2pi.")
np.savez("Paper_Data/individual_pulses.npz",thetas=thetas,
         **{f"{m}_{q}":res[m][q] for m in res for q in ['P','F']})

fig,ax=plt.subplots(1,2,figsize=(14,5))
cols={'bare':'cornflowerblue','phys':'seagreen','ideal':'firebrick'}
labs={'bare':'bare (no GCR)','phys':'physical GCR (correct axis)','ideal':'ideal vec*vecf'}
for m in ['bare','phys','ideal']:
    ax[0].plot(thetas/np.pi,1-res[m]['F'],'-o',ms=4,color=cols[m],label=labs[m])
    ax[1].plot(thetas/np.pi,1-res[m]['P'],'-o',ms=4,color=cols[m],label=labs[m])
for a,t in zip(ax,['success infidelity  $1-F_H$','failure probability  $1-P_{succ}$']):
    for xa in [0.5,1.0,2.0]: a.axvline(xa,ls=':',color='gray',alpha=0.6)
    a.set_yscale('log'); a.set_xlabel(r'pulse angle $\theta/\pi$'); a.set_title(t)
    a.grid(alpha=0.3); a.legend(fontsize=9)
ax[0].text(2.02,ax[0].get_ylim()[1]*0.2,'2π',fontsize=9,color='gray')
fig.suptitle('Exact BB1 pulses run individually (correct axis): fidelity & success prob vs angle',fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/individual_pulses.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/individual_pulses.png")
