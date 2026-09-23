"""Add the two prepended-momentum-BLOCK GCR-BB1 schemes (variant A: free block; variant B: front-loaded
block -- both from run_prepended_block_compare.py's Paper_Data/prepended_block_params.npz) to the noise
analysis, matching run_gcrbb1_noise.py's treatment of the single-pre-correction GCR-BB1 scheme. Computes
BOTH, for each block variant:
  * readout error P(e|eps)      -> merged as 'GCR-BB1-block{A,B}|{cond}|{pan}|q'       into mesolve_4.npz
  * fixed-beta_corr fidelity 1-F-> merged as 'GCR-BB1-block{A,B}|{cond}|{pan}|corr_fid' into corr_disp_fidelity.npz
Same noise model / rates / conventions as run_mesolve_4.py + run_gcrbb1_noise.py (Ncav=400, NE=8 default).
Block segment order (time order) = C0,C1,C2,C3 then the four BARE BB1 kicks K0,K1,K2,K3 -- this matches
Uof_block = BB1U*Cblock in run_prepended_block_compare.py (Cblock composes C0 first; BB1U composes K0
first). The terminal BB1 kick (theta=pi/2) IS the readout target rotation; no extra target is appended.
The calibration displacement D used for corr_fid is recomputed PER SCHEME (each block has its own
noiseless zero-eps back-action), exactly as calibrate_D() in run_gcrbb1_noise.py.
Rates: gamma=gamma_phi=1/200, kappa=1/1000 us^-1.
--test: does NOT touch Paper_Data; writes /tmp/block_noise_test.npz and prints keys+shapes instead."""
import sys, time, numpy as np
from qutip import *
t0=time.time(); TEST='--test' in sys.argv
_pos=[a for a in sys.argv[1:] if a!='--test']
Ncav=int(_pos[0]) if len(_pos)>0 else 400
NE=int(_pos[1]) if len(_pos)>1 else 8
gam=1/200.; gamphi=1/200.; kappa=1/1000.
aC=destroy(Ncav); xO=(aC+aC.dag())/np.sqrt(2); pO=(-1j)*(aC-aC.dag())/np.sqrt(2)
def SIG(p): return np.cos(p)*sigmax()+np.sin(p)*sigmay()
gq=basis(2,0); py=(basis(2,0)+1j*basis(2,1)).unit(); pydm=ket2dm(py)
Delta=0.34; ep=Delta**2; r=-np.log(Delta); sq=np.sqrt(np.pi); phi1=np.arccos(-1/8)
def logical(mu):
    psi=0*basis(Ncav); a=np.sqrt(np.pi/2); nmax=int(np.sqrt(2*Ncav)/(2*a))+2
    for n in range(-nmax,nmax+1):
        psi=psi+np.exp(-((2*n+mu)*a*Delta)**2/2)*displace(Ncav,(2*n+mu)*a)*squeeze(Ncav,r)*basis(Ncav,0)
    return psi.unit()
L0=logical(0); u=np.sqrt(np.pi)/(2*np.sqrt(2)); shift=np.sqrt(np.pi/2)/2
_blk=np.load('Paper_Data/prepended_block_params.npz')
paramsA=_blk['variantA_params']; paramsB=_blk['variantB_params']
print(f"[{time.time()-t0:.0f}s] Ncav={Ncav} NE={NE} TEST={TEST} block params A={np.round(paramsA,4)} "
      f"B={np.round(paramsB,4)}",flush=True)
BB1=[(np.pi,phi1),(2*np.pi,3*phi1),(np.pi,phi1),(np.pi/2,0.0)]
def kick(i): th,ph=BB1[i]; return (th/sq)*tensor(xO,SIG(ph)), th/np.sqrt(2*np.pi)
def det_corr(c,nu): return c*tensor(pO,SIG(nu)), abs(c)/np.sqrt(2)
def U(K): return (-1j*K).expm(method='dense')
def sc_block(p):
    L=[]
    for k in range(4): Kc,tc=det_corr(p[2*k],p[2*k+1]); L.append(('U',U(Kc),Kc,tc))  # C0,C1,C2,C3
    for i in range(4): Kk,tk=kick(i); L.append(('U',U(Kk),Kk,tk))                    # then bare K0..K3
    return L
SCHEMES={'GCR-BB1-blockA':sc_block(paramsA),'GCR-BB1-blockB':sc_block(paramsB)}
print(f"[{time.time()-t0:.0f}s] schemes built ({[(k,len(v)) for k,v in SCHEMES.items()]})",flush=True)
sm=tensor(qeye(Ncav),sigmam()); sz=tensor(qeye(Ncav),sigmaz()); aa=tensor(aC,qeye(2))
COPS={'noiseless':None,
      'complete':[np.sqrt(gam)*sm,np.sqrt(gamphi/2)*sz,np.sqrt(kappa)*aa],
      'tr_decay':[np.sqrt(gam)*sm],'tr_deph':[np.sqrt(gamphi/2)*sz],'osc_decay':[np.sqrt(kappa)*aa]}
opts=Options(nsteps=20000,atol=1e-8,rtol=1e-6)
def run(scheme,beta,cond):
    psi=(displace(Ncav,beta)*L0).unit(); pdm=ket2dm(psi)
    rho=tensor(pdm,ket2dm(gq)); cops=COPS[cond]
    for kind,op,K,t in scheme:
        if cops is None: rho=op*rho*op.dag()
        else: rho=mesolve(K/t,rho,[0,t],c_ops=cops,options=opts).states[-1]
    return psi,rho
def calibrate_D(scheme):
    _,rho=run(scheme,0.0,'noiseless'); ro=rho.ptrace(0)
    bx=float(np.real(expect(xO,ro)))/np.sqrt(2); bp=float(np.real(expect(pO,ro)))/np.sqrt(2)
    corr=1j*(bp+np.sqrt(np.pi/2)/2)-bx
    return displace(Ncav,corr),bx,bp,corr
epsA=np.linspace(0,u,NE); epsB=np.linspace(2*u,3*u,NE)
conds=['noiseless','complete','tr_decay','tr_deph','osc_decay']
q_out={}; f_out={}
for scn,scheme in SCHEMES.items():
    D,bx,bp,corr=calibrate_D(scheme)
    print(f"[{time.time()-t0:.0f}s] {scn} calib: back_x={bx:.4f} back_p={bp:.4f} corr={corr:.4f}",flush=True)
    for cond in conds:
        for pan,eps_arr,flipq in [('a',epsA,False),('b',epsB,True)]:
            qs=[]; fs=[]
            for e in eps_arr:
                psi,rho=run(scheme,e-shift,cond)
                qp=1-float(np.real(expect(pydm,rho.ptrace(1))))
                ro=rho.ptrace(0); fc=float(fidelity(ket2dm(psi),D*ro*D.dag()))
                qs.append((1-qp) if flipq else qp); fs.append(fc)
            q_out[f'{scn}|{cond}|{pan}|q']=np.array(qs)
            f_out[f'{scn}|{cond}|{pan}|corr_fid']=np.array(fs)
            print(f"[{time.time()-t0:.0f}s] {scn:18s} {cond:10s} {pan}  q:[{qs[0]:.3e}..{qs[-1]:.3e}] "
                  f"corr_fid:[{fs[0]:.4f}..{fs[-1]:.4f}]",flush=True)
if TEST:
    combined=dict(q_out); combined.update(f_out)
    np.savez('/tmp/block_noise_test.npz',**combined)
    print(f"[{time.time()-t0:.0f}s] TEST mode: wrote /tmp/block_noise_test.npz ({len(combined)} keys), "
          f"Paper_Data untouched",flush=True)
    for k,v in combined.items():
        arr=np.asarray(v)
        print(f"   {k:35s} shape={arr.shape} finite={np.all(np.isfinite(arr))}",flush=True)
else:
    for path,new in [('Paper_Data/mesolve_4.npz',q_out),('Paper_Data/corr_disp_fidelity.npz',f_out)]:
        d=dict(np.load(path)); d.update(new); np.savez(path,**d)
        print(f"[{time.time()-t0:.0f}s] merged {len(new)} keys into {path}",flush=True)
print(f"[{time.time()-t0:.0f}s] DONE",flush=True)
