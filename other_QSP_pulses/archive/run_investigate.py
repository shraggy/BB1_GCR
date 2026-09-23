"""
Reconcile: reproduce the EXACT notebook GCR_BB1 (cell 8f78613c: vec*vecf on the 3 correction
pulses, vec on the final target pulse) and compare its readout P(+1) to
  - all-vec*vecf (my 'ideal', red)
  - all-physical correct-axis (my 'gray, no fix')
  - bare BB1
at fine resolution, printing P near integer multiples of sqrt(pi).
"""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=140
aOp=destroy(Ncav)
isq=np.sqrt(2); xOp=(aOp+aOp.dag())/isq; pOp=(-1j)*(aOp-aOp.dag())/isq
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
def sigma_xyz(th,phi): return np.cos(th)*SZ+np.sin(th)*sigma(phi)
def rot_xy(th,phi): return (-1j*th/2*(np.cos(phi)*SX+np.sin(phi)*SY)).expm(method='dense')
def vec_f(v,R): return R*v*R.dag()
def svec(v): return v[0]*SX+v[1]*SY+v[2]*SZ
def rot_bloch(n,ax,ang):
    k=np.array(ax,float); k=k/np.linalg.norm(k); n=np.array(n,float)
    return n*np.cos(ang)+np.cross(k,n)*np.sin(ang)+k*np.dot(k,n)*(1-np.cos(ang))

Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi)
theta=np.pi/2; phi0=0; phi1=np.arccos(-theta/(4*np.pi)); x=np.pi/theta
beta=(theta/2/(np.sqrt(np.pi)/2))

def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy(th,axis): return tensor(-1j*(th*ep/sq)*pOp,axis).expm(method='dense')

# ---- EXACT notebook GCR_BB1 (cell 8f78613c) ----
def GCR_BB1_notebook(initial):
    state=tensor(initial,g)
    vecf=vec_f(sigma_xyz(0,0),rot_xy(0,0)); vec=1j*sigma(phi1)*vecf
    Op4=tensor(-1j*x*beta*(xOp),sigma(phi1)).expm(method='dense')*tensor(-1j*x*beta*(ep*pOp),vec*vecf).expm(method='dense')
    state=(Op4*state).unit()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(3*phi1)*vecf
    Op3=tensor(-2j*np.sqrt(np.pi)*(xOp),sigma(3*phi1)).expm(method='dense')*tensor(-2j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    state=(Op3*state).unit()
    vecf=vec_f(vecf,rot_xy(4*np.pi,3*phi1)); vec=1j*sigma(phi1)*vecf
    Op2=tensor(-1j*np.sqrt(np.pi)*(xOp),sigma(phi1)).expm(method='dense')*tensor(-1j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(method='dense')
    state=(Op2*state).unit()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(phi0)*vecf
    Op1=tensor(-1j*beta*(xOp),sigma(phi0)).expm(method='dense')*tensor(-1j*beta*(ep*pOp),vec).expm(method='dense')
    state=(Op1*state).unit()
    return state

BB1seq=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def build_correct(mode):  # 'ideal' (all vec*vecf) or 'phys' (all correct-axis)
    ops=[]; n=np.array([0,0,1.0])
    for th,phi in BB1seq:
        ox=Opx(th,phi); phiv=np.array([np.cos(phi),np.sin(phi),0.0])
        if mode=='ideal': ops.append(ox*Opy(th,1j*sigma(phi)))
        elif mode=='bare': ops.append(ox)
        else:
            gam=np.cross(n,phiv); ops.append(ox*Opy(th,svec(gam/np.linalg.norm(gam))))
        n=rot_bloch(n,phiv,th)
    return ops
def run(ops,init):
    s=tensor(init,g)
    for U in ops: s=(U*s).unit()
    return s

N=81; alph=np.linspace(-2*sq,2*sq,N); a=-sq/2; m=alph/sq
P={'notebook':[],'ideal':[],'phys':[],'bare':[]}
Sideal=build_correct('ideal'); Sphys=build_correct('phys'); Sbare=build_correct('bare')
for al in alph:
    alpha=al+a
    initial=(displace(Ncav,alpha/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit()
    P['notebook'].append(np.abs(expect(ket2dm(py),GCR_BB1_notebook(initial).ptrace(1))))
    P['ideal'].append(np.real(expect(ket2dm(py),run(Sideal,initial).ptrace(1))))
    P['phys'].append(np.real(expect(ket2dm(py),run(Sphys,initial).ptrace(1))))
    P['bare'].append(np.real(expect(ket2dm(py),run(Sbare,initial).ptrace(1))))
for k in P: P[k]=np.array(P[k])

print("m=<x-axis>=alph/sqrt(pi);  P(+1) near integer m (the integer-sqrt(pi) points):")
print(f"{'m':>7}{'notebook':>10}{'ideal':>9}{'phys':>9}{'bare':>9}")
for i in range(N):
    if abs(m[i]-round(m[i]))<0.03 or abs(abs(m[i]%1)-0.5)<0.03:
        print(f"{m[i]:>7.2f}{P['notebook'][i]:>10.3f}{P['ideal'][i]:>9.3f}{P['phys'][i]:>9.3f}{P['bare'][i]:>9.3f}")
print(f"\nmax|notebook-ideal|={np.max(np.abs(P['notebook']-P['ideal'])):.3f}  "
      f"max|notebook-phys|={np.max(np.abs(P['notebook']-P['phys'])):.3f}")

plt.figure(figsize=(10,6))
plt.plot(m,P['notebook'],'o-',color='purple',ms=3,lw=1.5,label='EXACT notebook GCR_BB1 (vec*vecf + last vec)')
plt.plot(m,P['ideal'],'--',color='firebrick',lw=2.4,label='all vec*vecf (ideal)')
plt.plot(m,P['phys'],':',color='0.4',lw=2.2,label='all physical correct-axis (no fix)')
plt.plot(m,P['bare'],color='black',lw=1.3,label='bare BB1')
for xi in [-2,-1,0,1,2]: plt.axvline(xi,ls=':',color='gray',alpha=0.4)
plt.xlabel(r'$\langle x\rangle$-axis $=\alpha_1/\sqrt{\pi}$'); plt.ylabel('P(+1)')
plt.title('Reconciliation: exact notebook BB1(GCR) vs ideal vs physical')
plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Paper_Figures/investigate.png",dpi=130,bbox_inches="tight")
print(f"\n[{time.time()-t0:.0f}s] saved Paper_Figures/investigate.png")
