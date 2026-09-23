"""Port the user's exact GCR_BB1 / BB1 (cell 12) and check what the 'BB1(GCR)' curve is:
washout (unitary ~0.31) or ideal (non-unitary ~4e-7)? Ncav=100 for speed, a few eps."""
import numpy as np
from qutip import *
g=basis(2,0); e=basis(2,1); py=(g+1j*e).unit()
Ncav=100
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def sigma_xyz(theta,phi): return np.cos(theta)*sigmaz()+np.sin(theta)*sigma(phi)
def rot_xy(theta,phi): return (-1j*theta/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm()
def vec_f(vec_i,rot): return rot*vec_i*rot.dag()
Delta=0.34; r=-np.log(Delta); theta=np.pi/2; phi0=0; phi1=np.arccos(-theta/(4*np.pi)); ep=Delta**2
x=np.pi/theta; beta=(theta/2/(np.sqrt(np.pi)/2))
def logical(mu,delta,Ncav,normalize=True):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2)
    for n in range(-np.int64(Ncav+mu/2),np.int64(Ncav+mu/2)-mu+1):
        psi=psi+np.exp(-((2*n+mu)*a*delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit() if normalize else psi
def GCR_BB1(initial):
    state=tensor(initial,g)
    vecf=vec_f(sigma_xyz(0,0),rot_xy(0,0)); vec=1j*sigma(phi1)*vecf
    Op4=tensor(-1j*x*beta*(xOp),sigma(phi1)).expm()*tensor(-1j*x*beta*(ep*pOp),vec*vecf).expm(); state=(Op4*state).unit()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(3*phi1)*vecf
    Op3=tensor(-2j*np.sqrt(np.pi)*(xOp),sigma(3*phi1)).expm()*tensor(-2j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(); state=(Op3*state).unit()
    vecf=vec_f(vecf,rot_xy(4*np.pi,3*phi1)); vec=1j*sigma(phi1)*vecf
    Op2=tensor(-1j*np.sqrt(np.pi)*(xOp),sigma(phi1)).expm()*tensor(-1j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm(); state=(Op2*state).unit()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(phi0)*vecf
    Op1=tensor(-1j*beta*(xOp),sigma(phi0)).expm()*tensor(-1j*beta*(ep*pOp),vec*vecf).expm(); state=(Op1*state).unit()
    return state
def BB1(initial):
    state=tensor(initial,g)
    for (coef,ph) in [(np.sqrt(np.pi),phi1),(2*np.sqrt(np.pi),3*phi1),(np.sqrt(np.pi),phi1),(beta,phi0)]:
        Op=tensor(1j*coef*(xOp),sigma(ph)).expm(); state=(Op.dag()*state).unit()
    return state
tgt=logical(0,Delta,Ncav)
print(f"{'eps':>6} {'GCR_BB1 P(+1)':>14} {'BB1 P(+1)':>11}")
for epsv in [0.0,0.3,0.62]:
    init=(displace(Ncav,epsv/np.sqrt(2))*tgt).unit()
    pg=float(np.abs(expect(ket2dm(py),GCR_BB1(init).ptrace(1))))
    pb=float(np.abs(expect(ket2dm(py),BB1(init).ptrace(1))))
    print(f"{epsv:>6} {pg:>14.4e} {pb:>11.4e}")

print("\nscan to locate peak (eps_plot = alph/sqrt2 + sqrt(pi/2)/2):")
print(f"{'alph':>8} {'eps_plot':>9} {'GCR_BB1 err':>12} {'BB1 err':>10}")
import numpy as np
for alph in np.linspace(-1.3,-0.4,10):
    init=(displace(Ncav,alph/np.sqrt(2))*tgt).unit()
    pg=float(np.abs(expect(ket2dm(py),GCR_BB1(init).ptrace(1))))
    pb=float(np.abs(expect(ket2dm(py),BB1(init).ptrace(1))))
    epl=alph/np.sqrt(2)+np.sqrt(np.pi/2)/2
    # error = 1-P(+1) near the no-error peak (P(+1)->1 there)
    print(f"{alph:>8.3f} {epl:>9.3f} {1-pg:>12.3e} {1-pb:>10.3e}")
