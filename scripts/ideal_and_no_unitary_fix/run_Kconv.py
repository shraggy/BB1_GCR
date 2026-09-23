"""Convergence of the green (correct-axis) split fix with K=1..16, at the operating points.
Left: P(-1) near x=0 (log). Right: P(+1) near x=sqrt(pi) (log). vs bare BB1 and ideal.
K=1 is the physical no-fix; larger K = smaller sub-angles."""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def corr_split(K):
    out=[]
    for i,(th,phi) in enumerate(BB1): out+=[(th,phi)] if i==3 else [(th/K,phi)]*K
    return out
def build(seq,mode):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        else:
            gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
N=121; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)
Ks=[1,2,4,8,12,16]
C={'bare':sweep(build(BB1,'bare')),'ideal':sweep(build(BB1,'ideal'))}
for K in Ks:
    C[f'K{K}']=sweep(build(corr_split(K),'correct')); print(f"[{time.time()-t0:.0f}s] K={K} done")
np.savez("Paper_Data/Kconv.npz",m=m,**C)
def clip(v): return np.clip(v,1e-7,None)
i0=int(np.argmin(np.abs(m))); i1=int(np.argmin(np.abs(m-1)))
print("\n K :  P(-1)@x=0   P(+1)@x=rt(pi)   (sub-angle of 2pi pulse)")
print(f" bare: {1-C['bare'][i0]:.4f}     {C['bare'][i1]:.4f}")
print(f"ideal: {1-C['ideal'][i0]:.2e}   {C['ideal'][i1]:.2e}")
for K in Ks:
    print(f" K={K:<2}: {1-C[f'K{K}'][i0]:.4f}     {C[f'K{K}'][i1]:.4f}       2pi/{K}={2*180//K} deg")

greens=plt.cm.Greens(np.linspace(0.35,1.0,len(Ks)))
fig,ax=plt.subplots(1,2,figsize=(13,5.4))
for k,P in [('bare','tab:blue'),('ideal','firebrick')]:
    ax[0].plot(m,clip(1-C[k]),('--' if k=='ideal' else '-'),color=P,lw=2.0,label=('ideal (non-unitary)' if k=='ideal' else 'bare BB1'))
    ax[1].plot(m,clip(C[k]),('--' if k=='ideal' else '-'),color=P,lw=2.0)
for j,K in enumerate(Ks):
    ax[0].plot(m,clip(1-C[f'K{K}']),color=greens[j],lw=1.8,label=f'K={K}')
    ax[1].plot(m,clip(C[f'K{K}']),color=greens[j],lw=1.8)
ax[0].set_yscale('log'); ax[0].set_xlim(-0.7,0.7); ax[0].set_ylim(1e-5,1)
ax[0].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[0].set_ylabel(r'$P(-1)$'); ax[0].grid(alpha=0.3)
ax[0].set_title(r'$P(-1)$ near $\langle x\rangle=0$'); ax[0].legend(fontsize=8,ncol=2)
ax[1].set_yscale('log'); ax[1].set_xlim(0.3,1.7); ax[1].set_ylim(1e-5,1)
ax[1].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[1].set_ylabel(r'$P(+1)$'); ax[1].grid(alpha=0.3)
ax[1].set_title(r'$P(+1)$ near $\langle x\rangle=\sqrt{\pi}$')
fig.suptitle('Green (correct-axis) split fix: convergence with K (K=1 is no fix)',fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/Kconv.png",dpi=130,bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved Paper_Figures/Kconv.png")
