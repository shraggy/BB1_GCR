"""Corrective-displacement-corrected oscillator fidelity (author's cell-14 recipe), replacing the
impurity/sBs oscillator story. For each of the four readouts (single GCR, deterministic BB1(GCR)
[QITE], bare BB1, heralded [herald-RUS, exact-M+renormalize]):
  1) calibrate ONE corrective displacement D on the NOISELESS no-error logical-0 readout output:
       back_x = <xO>(perfect_osc)/sqrt(2);  back_p = <pO>(perfect_osc)/sqrt(2)
       corr = +1j*(back_p + sqrt(pi/2)/2) - back_x ;  D = displace(Ncav, corr)
  2) apply that SAME D to the readout's output oscillator state at every eps in the logical-0
     window [0,u], under noiseless and 'complete' noise, and report fidelity(ket2dm(input), D rho D^)
     vs input (the displacement-error-bearing input state actually fed into the scheme), qutip's
     fidelity() (non-squared), matching the paper's exact convention.
  3) also report the UNCORRECTED fidelity for reference/contrast.
Reuses the exact scheme definitions (kick/det_corr/Mexact/sc_gcr/sc_bb1/sc_qite/sc_herald) and noise
model (COPS, rates) from run_mesolve_4.py.
"""
import sys, time, numpy as np
from qutip import *
t0 = time.time()
Ncav = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
gam = 1/200.; gamphi = 1/200.; kappa = 1/1000.
aC = destroy(Ncav); xO = (aC+aC.dag())/np.sqrt(2); pO = (-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq = basis(2,0); py = (basis(2,0)+1j*basis(2,1)).unit(); pydm = ket2dm(py)
Delta = 0.34; ep = Delta**2; r = -np.log(Delta); sq = np.sqrt(np.pi); phi1 = np.arccos(-1/8)
def logical(mu):
    psi = 0*basis(Ncav); a = np.sqrt(np.pi/2); nmax = int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax, nmax+1):
        psi = psi + np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0 = logical(0); u = np.sqrt(np.pi)/(2*np.sqrt(2)); shift = np.sqrt(np.pi/2)/2
qp = np.load('Paper_Data/qite_det_params.npy')
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} NE={NE} QITE params: {np.round(qp,3)}", flush=True)
BB1 = [(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph = BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def Mexact(i): th,ph = BB1[i]; return (tensor((th*ep/sq)*pO,SIG(ph))).expm(method='dense')
def U(K): return (-1j*K).expm(method='dense')
def sc_gcr():
    K,t = det_corr((np.pi/2)*ep/sq, 0+np.pi/2); Kk,tk = kick(3)
    return [('U',U(K),K,t),('U',U(Kk),Kk,tk)]
def sc_bb1():
    L = []
    for i in range(4): Kk,tk = kick(i); L.append(('U',U(Kk),Kk,tk))
    return L
def sc_qite():
    L = []
    for i in range(4):
        K,t = det_corr(qp[2*i],qp[2*i+1]); Kk,tk = kick(i)
        L += [('U',U(K),K,t),('U',U(Kk),Kk,tk)]
    return L
def sc_herald():
    L = []
    for i in range(4):
        Kk,tk = kick(i)
        if i < 3: L.append(('M',Mexact(i),None,0.0))
        else: K,t = det_corr((np.pi/2)*ep/sq,0+np.pi/2); L.append(('U',U(K),K,t))
        L.append(('U',U(Kk),Kk,tk))
    return L
SCHEMES = {'GCR':sc_gcr(),'BB1(GCR)':sc_qite(),'BB1':sc_bb1(),'heralded':sc_herald()}
print(f"[{time.time()-t0:.0f}s] schemes built", flush=True)
sm = tensor(qeye(Ncav),sigmam()); sz = tensor(qeye(Ncav),sigmaz()); aa = tensor(aC,qeye(2))
COPS = {'noiseless':None,
        'complete':[np.sqrt(gam)*sm, np.sqrt(gamphi/2)*sz, np.sqrt(kappa)*aa],
        'tr_decay':[np.sqrt(gam)*sm], 'tr_deph':[np.sqrt(gamphi/2)*sz], 'osc_decay':[np.sqrt(kappa)*aa]}
opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)

def run_scheme(scheme, beta, cond):
    psi = (displace(Ncav,beta)*L0).unit(); pdm = ket2dm(psi)
    rho = tensor(pdm, ket2dm(gq)); cops = COPS[cond]
    for kind,op,K,t in scheme:
        if kind == 'M':
            rho = op*rho*op.dag(); rho = rho/rho.tr()
        elif cops is None:
            rho = op*rho*op.dag()
        else:
            rho = mesolve(K/t, rho, [0,t], c_ops=cops, options=opts).states[-1]
    return psi, rho.ptrace(0)

def calibrate_D(scheme):
    psi0, ro0 = run_scheme(scheme, 0.0, 'noiseless')   # perfect = readout(logical(0)), no error
    back_x = float(np.real(expect(xO, ro0)))/np.sqrt(2)
    back_p = float(np.real(expect(pO, ro0)))/np.sqrt(2)
    corr = 1j*(back_p + np.sqrt(np.pi/2)/2) - back_x
    return displace(Ncav, corr), back_x, back_p, corr

epsA = np.linspace(0, u, NE); epsB = np.linspace(2*u, 3*u, NE)
WINDOWS = [('a', epsA), ('b', epsB)]
CONDS = ['noiseless','complete','tr_decay','tr_deph','osc_decay']
results = {}
for scn, scheme in SCHEMES.items():
    D, bx, bp, corr = calibrate_D(scheme)   # ONE fixed corrective displacement, calibrated on noiseless logical-0
    print(f"[{time.time()-t0:.0f}s] {scn}: back_x={bx:.4f} back_p={bp:.4f} corr={corr:.4f}", flush=True)
    for cond in CONDS:
        for pan, eps_arr in WINDOWS:
            raws = []; corrs = []
            for eps in eps_arr:
                beta = eps - shift
                psi, ro = run_scheme(scheme, beta, cond)
                f_raw = float(fidelity(ket2dm(psi), ro))
                ro_corr = D*ro*D.dag()
                f_corr = float(fidelity(ket2dm(psi), ro_corr))
                raws.append(f_raw); corrs.append(f_corr)
            raws = np.array(raws); corrs = np.array(corrs)
            results[f'{scn}|{cond}|{pan}|raw_fid'] = raws
            results[f'{scn}|{cond}|{pan}|corr_fid'] = corrs
            print(f"[{time.time()-t0:.0f}s] {scn:10s} {cond:10s} {pan} "
                  f"corr_fid: [{corrs[0]:.4f} .. {corrs[-1]:.4f}]", flush=True)
            np.savez('Paper_Data/corr_disp_fidelity.npz', epsA=epsA, epsB=epsB, u=u, **results)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
