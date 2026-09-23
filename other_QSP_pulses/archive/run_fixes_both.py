"""
Apply the splitting fix to BOTH physical constructions from bb1_axiscompare.png:
  green = correct axis   sigma_gamma = n_hat x phi_hat  (n_hat rotates by theta)
  gray  = notebook 'vec' = 1j sigma(phi) vecf           (vecf rotates by 2*theta)
Show two log panels vs bare BB1 and the ideal (non-unitary):
  left : P(-1) near x=0  (even-bin operating point, target P(+1)=1)
  right: P(+1) near x=1  (odd-bin  operating point, target P(+1)=0)
Curves: bare, ideal, green no-fix, gray no-fix, green split-fix K12, gray split-fix K12.
"""
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
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def vec_f(v,R): return R*v*R.dag()
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
def build(seq,mode):  # mode: 'bare','ideal','correct','notebook'
    ops=[]; n=np.array([0,0,1.0]); vecf=SZ
    for th,phi in seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='bare': ops.append(ox)
        elif mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        elif mode=='correct':
            gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        else: # notebook vec
            ops.append(ox*Opy(th,1j*sigma(phi)*vecf))
        n=rot_bloch(n,phiv,th); vecf=vec_f(vecf,rot_xy(2*th,phi))
    return ops

N=161; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
inits=[(displace(Ncav,(al+a)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for al in alph]
def sweep(ops):
    out=[]
    for it in inits:
        s=tensor(it,g)
        for U in ops: s=(U*s).unit()
        out.append(np.real(expect(ket2dm(py),s.ptrace(1))))
    return np.array(out)

C={}
C['bare']=sweep(build(BB1,'bare')); print(f"[{time.time()-t0:.0f}s] bare")
C['ideal']=sweep(build(BB1,'ideal')); print(f"[{time.time()-t0:.0f}s] ideal")
C['green_nofix']=sweep(build(BB1,'correct')); print(f"[{time.time()-t0:.0f}s] green nofix")
C['gray_nofix']=sweep(build(BB1,'notebook')); print(f"[{time.time()-t0:.0f}s] gray nofix")
C['green_fix']=sweep(build(corr_split(12),'correct')); print(f"[{time.time()-t0:.0f}s] green fix")
C['gray_fix']=sweep(build(corr_split(12),'notebook')); print(f"[{time.time()-t0:.0f}s] gray fix")
np.savez("Paper_Data/fixes_both.npz",m=m,**C)

def clip(v): return np.clip(v,1e-7,None)
sty={'bare':('tab:blue','-',2.0,'bare BB1'),
     'ideal':('firebrick','--',1.8,'ideal (non-unitary)'),
     'green_nofix':('#2ca02c','-',2.0,'no fix: correct axis $\\hat n\\!\\times\\!\\hat\\phi$'),
     'gray_nofix':('0.4','-',2.0,'no fix: notebook vec'),
     'green_fix':('#2ca02c',':',2.2,'fix (split K12): correct axis'),
     'gray_fix':('0.4',':',2.2,'fix (split K12): notebook vec')}
fig,ax=plt.subplots(1,2,figsize=(13,5.4))
for k,(c,ls,lw,lab) in sty.items():
    ax[0].plot(m,clip(1-C[k]),ls,color=c,lw=lw,label=lab)
    ax[1].plot(m,clip(C[k]),ls,color=c,lw=lw,label=lab)
ax[0].set_yscale('log'); ax[0].set_xlim(-0.7,0.7); ax[0].set_ylim(1e-5,1)
ax[0].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[0].set_ylabel(r'$P(-1)$')
ax[0].set_title(r'$P(-1)$ near $\langle x\rangle=0$ (even bin: target $+1$)'); ax[0].grid(alpha=0.3)
ax[1].set_yscale('log'); ax[1].set_xlim(0.3,1.7); ax[1].set_ylim(1e-5,1)
ax[1].set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax[1].set_ylabel(r'$P(+1)$')
ax[1].set_title(r'$P(+1)$ near $\langle x\rangle=1$ (odd bin: target $-1$)'); ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=8,loc='lower center')
fig.suptitle('Fixes applied to both physical BB1(GCR) constructions, at the operating points',fontsize=12)
fig.tight_layout(); fig.savefig("Paper_Figures/fixes_both.png",dpi=130,bbox_inches="tight")
i0=int(np.argmin(np.abs(m))); i1=int(np.argmin(np.abs(m-1)))
print("\nP(-1) at x=0:", {k:round(float(1-C[k][i0]),5) for k in C})
print("P(+1) at x=1:", {k:round(float(C[k][i1]),5) for k in C})
print(f"[{time.time()-t0:.0f}s] saved Paper_Figures/fixes_both.png")
