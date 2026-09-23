"""Two more splitting schemes (correct axis n x phi), vs bare & ideal:
 (a) split ALL four pulses (including the final pi/2 target)
 (b) split ONLY the middle 2pi pulse (keep pi, pi, pi/2 whole)
Compare with the earlier scheme (split first three, keep target). Kfull-style figures +
P(-1)@x=0 convergence table."""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]  # idx 0..3; idx1=2pi, idx3=target
def split_scheme(scheme,K):
    out=[]
    for i,(th,phi) in enumerate(BB1):
        if scheme=='all': out+=[(th/K,phi)]*K
        elif scheme=='2pi': out+= ([(th/K,phi)]*K if i==1 else [(th,phi)])
        elif scheme=='first3': out+= ([(th,phi)] if i==3 else [(th/K,phi)]*K)
    return out
def build(seq):
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
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
d=np.load("Paper_Data/Kconv.npz"); bare=d['bare']; ideal=d['ideal']
i0=int(np.argmin(np.abs(m)))
Ks=[1,2,4,8,16]
data={'all':{},'2pi':{}}
for scheme in ['all','2pi']:
    for K in Ks:
        data[scheme][K]=sweep(build(split_scheme(scheme,K))); print(f"[{time.time()-t0:.0f}s] {scheme} K={K}")
np.savez("Paper_Data/split_variants.npz",m=m,bare=bare,ideal=ideal,
         **{f"{s}_K{K}":data[s][K] for s in data for K in Ks})

# convergence table
print("\nP(-1)@x=0:  bare=%.4f  ideal=%.2e"%(1-bare[i0],1-ideal[i0]))
print(f"{'K':>3} {'(a) split-all':>14} {'(b) split-2pi-only':>18} {'first3(prev)':>13}")
for K in Ks:
    f3 = 1-d[f'K{K}'][i0] if f'K{K}' in d.files else np.nan
    print(f"{K:>3} {1-data['all'][K][i0]:>14.4f} {1-data['2pi'][K][i0]:>18.4f} {f3:>13.4f}")

def clip(v): return np.clip(v,1e-7,None)
greens=plt.cm.Greens(np.linspace(0.4,1.0,len(Ks)))
for scheme,fname,ttl in [('all',"Paper_Figures/split_all.png","(a) split ALL 4 pulses (incl. target)"),
                         ('2pi',"Paper_Figures/split_2pi.png","(b) split ONLY the middle 2pi pulse")]:
    fig=plt.figure(figsize=(12.5,5.2)); gs=GridSpec(1,2,width_ratios=[2.0,1.0],wspace=0.26)
    ax=fig.add_subplot(gs[0]); axr=fig.add_subplot(gs[1])
    for j,K in enumerate(Ks): ax.plot(m,data[scheme][K],color=greens[j],lw=1.7,label=f'K={K}')
    ax.plot(m,ideal,color='firebrick',lw=2.2,ls='--',label='ideal (non-unitary)',zorder=9)
    ax.plot(m,bare,color='dodgerblue',lw=3.2,label='bare BB1',zorder=10)
    ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8,loc='lower center',ncol=3); ax.set_title('readout, full range'); ax.axvspan(-0.5,0.5,color='0.93',zorder=0)
    for j,K in enumerate(Ks): axr.plot(m,clip(1-data[scheme][K]),color=greens[j],lw=1.7)
    axr.plot(m,clip(1-ideal),color='firebrick',lw=2.2,ls='--',zorder=9)
    axr.plot(m,clip(1-bare),color='dodgerblue',lw=3.2,zorder=10)
    axr.set_yscale('log'); axr.set_xlim(-0.5,0.5); axr.set_ylim(1e-6,1)
    axr.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); axr.set_ylabel(r'$P(-1)$'); axr.grid(alpha=0.3)
    axr.set_title(r'$P(-1)$ near $\langle x\rangle=0$ (log)')
    fig.suptitle(ttl,fontsize=13); fig.tight_layout()
    fig.savefig(fname,dpi=130,bbox_inches="tight"); print("saved",fname)
print(f"[{time.time()-t0:.0f}s] done")
