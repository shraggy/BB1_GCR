"""Validation requested by coordinator, BEFORE any Ncav=400 runs:
1) Performance: precomputed/cached sbs_lib gates -> a 10-round noiseless round_x sBs sequence at
   Ncav=100 should run in seconds, not minutes.
2) Premise: does round_x ALONE (repeated, noiseless) reduce the impurity of an already-MIXED
   state (photon-loss-decohered logical(0), impurity ~0.05-0.1), i.e. does it autonomously purify
   toward the code, or does it only correct coherent displacement (leaving incoherent mixedness
   untouched or worse)?
"""
import time, numpy as np
from qutip import *
import sbs_lib as sb

Ncav = 100
t0 = time.time()
L0 = sb.logical(0, Ncav)
rho0 = ket2dm(L0)
print(f"[{time.time()-t0:.2f}s] ideal logical(0) impurity = {sb.impurity(rho0):.4e}")

# --- decohere via photon loss until impurity lands in [0.05,0.1] ---
aC = destroy(Ncav)
kappa_test = 0.05
dt = 0.05
rho = rho0
tt = 0.0
while sb.impurity(rho) < 0.07:
    rho = mesolve(0*qeye(Ncav), rho, [0, dt], c_ops=[np.sqrt(kappa_test)*aC], options=sb.opts).states[-1]
    tt += dt
    if tt > 50:
        break
im_decohered = sb.impurity(rho)
print(f"[{time.time()-t0:.2f}s] decohered state: t_loss={tt:.1f}, impurity={im_decohered:.4e}")

# --- performance check: time a single 10-round noiseless round_x sequence ---
t1 = time.time()
trace, rho_final, minimp = sb.run_sbs_sequence(rho, 'noiseless', Ncav, nrounds=10, early_stop=False)
elapsed = time.time() - t1
print(f"\n[PERF] 10-round noiseless round_x sequence elapsed = {elapsed:.2f}s (target: a few seconds)")
print(f"[PREMISE] impurity trace (decohered mixed state, round_x x10, noiseless):")
for i, v in enumerate(trace):
    print(f"  round {i:2d}: impurity = {v:.6e}")
print(f"[PREMISE] min impurity reached = {minimp:.6e} (started at {trace[0]:.6e})")
monotonic = all(trace[i+1] <= trace[i] + 1e-9 for i in range(len(trace)-1))
print(f"[PREMISE] monotonically non-increasing: {monotonic}")

# --- also check: does round_x still behave cleanly on the IDEAL codeword (near-identity)? ---
t2 = time.time()
trace_ideal, _, _ = sb.run_sbs_sequence(rho0, 'noiseless', Ncav, nrounds=5, early_stop=False)
print(f"\n[{time.time()-t2:.2f}s] ideal-codeword sanity trace (5 rounds): {['%.2e'%v for v in trace_ideal]}")

print(f"\n[TOTAL {time.time()-t0:.2f}s] DONE")
