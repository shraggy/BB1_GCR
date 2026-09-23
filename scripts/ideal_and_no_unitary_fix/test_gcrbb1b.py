"""Build GCR_BB1 operator ONCE (eps-independent), apply across a scan. Ncav=90, small logical sum."""
import numpy as np
from qutip import *
g=basis(2,0); e=basis(2,1); py=(g+1j*e).unit()
Ncav=90
aOp=destroy(Ncav); xOp=(aOp+aOp.dag())/np.sqrt(2); pOp=(-1j)*(aOp-aOp.dag())/np.sqrt(2)
def sigma(phi): return np.cos(phi)*sigmax()+np.sin(phi)*sigmay()
def sigma_xyz(theta,phi): return np.cos(theta)*sigmaz()+np.sin(theta)*sigma(phi)
def rot_xy(theta,phi): return (-1j*theta/2*(np.cos(phi)*sigmax()+np.sin(phi)*sigmay())).expm()
def vec_f(v,rot): return rot*v*rot.dag()
Delta=0.34; r=-np.log(Delta); theta=np.pi/2; phi0=0; phi1=np.arccos(-theta/(4*np.pi)); ep=Delta**2
x=np.pi/theta; beta=(theta/2/(np.sqrt(np.pi)/2))
def logical(mu,delta,Ncav):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2)
    for n in range(-6,7):
        psi=psi+np.exp(-((2*n+mu)*a*delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
# build GCR_BB1 unitary ONCE (operators are eps-independent)
def U_GCR_BB1():
    vecf=vec_f(sigma_xyz(0,0),rot_xy(0,0)); vec=1j*sigma(phi1)*vecf
    Op4=tensor(-1j*x*beta*(xOp),sigma(phi1)).expm()*tensor(-1j*x*beta*(ep*pOp),vec*vecf).expm()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(3*phi1)*vecf
    Op3=tensor(-2j*np.sqrt(np.pi)*(xOp),sigma(3*phi1)).expm()*tensor(-2j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm()
    vecf=vec_f(vecf,rot_xy(4*np.pi,3*phi1)); vec=1j*sigma(phi1)*vecf
    Op2=tensor(-1j*np.sqrt(np.pi)*(xOp),sigma(phi1)).expm()*tensor(-1j*np.sqrt(np.pi)*(ep*pOp),vec*vecf).expm()
    vecf=vec_f(vecf,rot_xy(2*np.pi,phi1)); vec=1j*sigma(phi0)*vecf
    Op1=tensor(-1j*beta*(xOp),sigma(phi0)).expm()*tensor(-1j*beta*(ep*pOp),vec*vecf).expm()
    return Op1*Op2*Op3*Op4
U=U_GCR_BB1(); tgt=logical(0,Delta,Ncav)
print("what is GCR_BB1 at its readout peak?")
print(f"{'alph':>8} {'eps_plot':>9} {'GCR_BB1 err(1-P+)':>18}")
for alph in np.linspace(-1.3,-0.5,9):
    init=(displace(Ncav,alph/np.sqrt(2))*tgt).unit()
    P=float(np.abs(expect(ket2dm(py),(U*tensor(init,g)).unit().ptrace(1))))
    print(f"{alph:>8.3f} {alph/np.sqrt(2)+np.sqrt(np.pi/2)/2:>9.3f} {1-P:>18.4e}")
