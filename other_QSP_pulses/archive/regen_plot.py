"""Regenerate the BB1(GCR) fixes plot with the NO-FIX (physical, unsplit) curve added.
Curves = different choices of the GCR pre-correction ('second pulse'):
  ideal (non-unitary), physical/no-fix (unitary correct axis), split (Fix1/4),
  amplitude-optimized (Fix2), and bare BB1 reference.
Saves Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/fixes_squarewave.png.
"""
import numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d=np.load("Paper_Data/fixes_final.npz")
m=d["m"]; Pbare=d["Pbare"]; Psplit=d["Psplit"]; Pfix2=d["Pfix2"]; Pideal=d["Pideal"]

# recompute the NO-FIX curve (physical, correct axis, unsplit) on the same grid
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')
phi1=np.arccos(-1/8); BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def build_phys():
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in BB1:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
N=len(m); a1=np.linspace(-2*sq,2*sq,N); aoff=-sq/2
inits=[(displace(Ncav,(x+aoff)/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit() for x in a1]
ops=build_phys(); Pnofix=[]
for it in inits:
    s=tensor(it,g)
    for U in ops: s=(U*s).unit()
    Pnofix.append(np.real(expect(ket2dm(py),s.ptrace(1))))
Pnofix=np.array(Pnofix)
Pideal_pl=Pideal; mask=np.abs(Pideal_pl-np.round(Pideal_pl))<0.07; tgt=np.round(Pideal_pl)
oe=lambda P: np.mean(np.abs(P-tgt)[mask])

fig,ax=plt.subplots(figsize=(9.5,6))
ax.plot(m,Pideal,color='firebrick',lw=2.8,ls='--',label=fr'ideal pre-correction $i\sigma_\phi$ (non-unitary): op {oe(Pideal):.3f}')
ax.plot(m,Psplit,color='#2ca02c',lw=2.8,label=fr'fix: split pre-correction (K=12): op {oe(Psplit):.3f}')
ax.plot(m,Pbare,color='black',lw=1.8,label=fr'bare BB1 (no GCR): op {oe(Pbare):.3f}')
ax.plot(m,Pnofix,color='0.55',lw=2.2,ls=':',label=fr'BB1(GCR), no fix (physical $\sigma_\gamma=\hat n\times\hat\phi$): op {oe(Pnofix):.3f}')
ax.set_xlabel(r'$\langle x\rangle/\sqrt{\pi}$'); ax.set_ylabel(r'$P(+1)$')
ax.set_title('BB1(GCR) modular readout: replacing the pre-correction (correct axis, noiseless)')
ax.legend(fontsize=8.5,loc='center'); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("Supp_Figures/BB1_GCR_fixes.pdf",bbox_inches="tight")
fig.savefig("Paper_Figures/fixes_squarewave.png",dpi=130,bbox_inches="tight")
np.savez("Paper_Data/fixes_final.npz",m=m,Pbare=Pbare,Psplit=Psplit,Pfix2=Pfix2,Pideal=Pideal,
         Pnofix=Pnofix,psucc=d["psucc"],lam=d["lam"])
print("op-err: ideal=%.3f split=%.3f bare=%.3f nofix=%.3f fix2=%.3f"%(oe(Pideal),oe(Psplit),oe(Pbare),oe(Pnofix),oe(Pfix2)))
print("saved Supp_Figures/BB1_GCR_fixes.pdf and Paper_Figures/fixes_squarewave.png")
