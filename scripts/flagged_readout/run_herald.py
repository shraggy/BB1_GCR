"""Heralded non-unitary fix: apply the ideal pre-correction at strength s (s=0 -> bare, unitary,
success=1; s=1 -> full ideal, non-unitary). For each s, compute the post-selected readout error
P(-1) at x=0 and the heralding success probability = ||M_s psi||^2 / lambda_max(M_s)^2
(block-encoding). Shows the tradeoff: reaching the ideal costs ~1e-4 success."""
import time, numpy as np
from qutip import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
g=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
SX,SY,SZ=sigmax(),sigmay(),sigmaz()
def sigma(phi): return np.cos(phi)*SX+np.sin(phi)*SY
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
I2=tensor(qeye(Ncav),qeye(2))
def Opx(th,phi): return tensor(-1j*(th/sq)*xOp,sigma(phi)).expm(method='dense')
def Opy_nonU(th,phi,s): return tensor(-1j*s*(th*ep/sq)*pOp,1j*sigma(phi)).expm(method='dense')
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def Mcomposite(s):  # unrenormalized non-unitary composite at strength s
    U=None
    for th,phi in BB1:
        op = Opx(th,phi)*Opy_nonU(th,phi,s)
        U = op if U is None else op*U
    return U
# input at x=0 (m=0 -> alph1=0 -> displacement (0-sqrt(pi)/2)/sqrt2)
a=-sq/2
psi0=tensor((displace(Ncav,a/np.sqrt(2))*(squeeze(Ncav,r)*basis(Ncav,0)).unit()).unit(), g)

ss=[0.0,0.25,0.5,0.75,1.0]
rows=[]
for s in ss:
    M=Mcomposite(s)
    lam=np.linalg.svd(M.full(),compute_uv=False)[0]
    out=(M*psi0)
    nrm2=float(out.norm()**2)
    st=out.unit()
    Pm1=float(1-np.real(expect(ket2dm(py),st.ptrace(1))))
    succ=nrm2/lam**2
    rows.append((s,Pm1,succ,lam))
    print(f"[{time.time()-t0:.0f}s] s={s}: readout P(-1)@0={Pm1:.2e}  lambda_max={lam:.2f}  success={succ:.2e}")

rows=np.array(rows)
plt.figure(figsize=(7.5,5.5))
plt.plot(rows[:,1],rows[:,2],'o-',color='purple',lw=2)
for s,Pm1,succ,lam in rows:
    plt.annotate(f's={s}',(Pm1,succ),textcoords="offset points",xytext=(6,6),fontsize=9)
plt.xscale('log'); plt.yscale('log'); plt.gca().invert_xaxis()
plt.xlabel(r'post-selected readout error $P(-1)$ at $x=0$  (better $\rightarrow$)')
plt.ylabel('heralding success probability')
plt.title('Heralded non-unitary fix: success vs readout error\n(s=0 bare/deterministic, s=1 full ideal)')
plt.grid(alpha=0.3,which='both'); plt.tight_layout()
plt.savefig("Paper_Figures/herald_tradeoff.png",dpi=130,bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved Paper_Figures/herald_tradeoff.png")
