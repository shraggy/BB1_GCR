# BB1(GCR) Readout: A Real-Noise Analysis

Companion repository to [`shraggy/NA-QSP_sims`](https://github.com/shraggy/NA-QSP_sims)
(simulations for [arXiv:2504.19992](https://arxiv.org/abs/2504.19992), *Non-Abelian
Quantum Signal Processing*).

That paper introduces the Gaussian-Controlled-Rotation (GCR) and shows that composing
four GCR pulses into the Abelian BB1 sequence — BB1(GCR) — should, in principle, give
a high-fidelity GKP modular readout. This repo asks whether that promise survives
once the pulses have to be physically realizable and the hardware is noisy. The
write-up is [`BB1GCR_note.tex`](BB1GCR_note.tex) / [`BB1GCR_note.pdf`](BB1GCR_note.pdf)
("Readout/Calibration of Erroneous GKP States"); everything else in this repo is the
code and data behind it.

## Environment

Same stack as the sibling repo: QuTiP, SciPy, NumPy, matplotlib. The notebook and
several scripts use the legacy `qutip.qip.operations` API (`rx`/`ry`/`rz`), which
requires **`qutip==4.7.6`** (not QuTiP 5.x) on **Python 3.9**. Create your own local
virtual environment rather than relying on any pre-built one:

```
python3.9 -m venv qutip-env
./qutip-env/bin/pip install qutip==4.7.6 qutip-qip numpy scipy matplotlib jupyter
```

## How to run

All scripts use relative paths (`Paper_Data/...`, `Paper_Figures/...`) and must be
run **from the repository root**, e.g.:

```
python scripts/damage_readout/run_damage_readout.py
python scripts/sbs_readout/run_task_A.py
```

## Repository layout

| Path | Purpose |
|---|---|
| [`BB1GCR_note.tex`](BB1GCR_note.tex), [`BB1GCR_note.pdf`](BB1GCR_note.pdf) | The document itself — the real-noise BB1(GCR) readout analysis. |
| [`Paper_Data/`](Paper_Data/) | Simulation outputs (`.npz`/`.npy`) underlying every figure in the note. |
| [`Paper_Figures/`](Paper_Figures/) | Main-text figures of the note. |
| [`Supp_Figures/`](Supp_Figures/) | Supplementary figures of the note. |
| [`fonts/`](fonts/) | `Roboto-Regular.ttf`, used by the plotting scripts for consistent figure typography. |
| [`notebooks/BB1_GCR_analysis.ipynb`](notebooks/BB1_GCR_analysis.ipynb) | The working notebook this analysis grew out of (builds on the sibling repo's `NA-QSP_sims.ipynb`; see its Fig. 2a / Fig. 14 cells for baseline GCR-vs-BB1 performance). |
| [`scripts/ideal_and_no_unitary_fix/`](scripts/ideal_and_no_unitary_fix/) | §"The ideal readout is non-unitary" / §"Why the physical (unitary) pulse fails" — establishes that the exact GCR pre-correction is non-unitary, and that no unitary variant rescues it. |
| [`scripts/flagged_readout/`](scripts/flagged_readout/) | §"The flagged readout" (+ heralding, restart/reuse, back-action, end-of-line sub-sections) — the post-selected modular readout construction. |
| [`scripts/calibration/`](scripts/calibration/) | §"Calibration: realizing the ideal correction, and its noise budget." |
| [`scripts/variational_compilation/`](scripts/variational_compilation/) | §"A deterministic readout by variational compilation, and the noise-limited verdict" — QITE-optimized BB1(GCR) alternative. |
| [`scripts/mesolve_noise/`](scripts/mesolve_noise/) | §"Readout under realistic noise: master-equation simulation" — the full-noise verdict. |
| [`scripts/damage_readout/`](scripts/damage_readout/) | §"Reading a pre-damaged codeword" — stress test on a codeword that idled under decay/dephasing before readout. |
| [`scripts/sbs_readout/`](scripts/sbs_readout/) | §"Reading an actively stabilized codeword" — stress test on a codeword stabilized by noisy small-big-small (SBS) rounds. Includes the shared `sbs_lib.py` module. |
| [`other_QSP_pulses/`](other_QSP_pulses/) | **Not referenced in the note.** Earlier exploratory work kept for provenance: `archive/` (superseded plotting/debugging scripts and console logs from before the analysis settled), `scrofulous_tycko/` (a SCROFULOUS/Tycko composite-pulse cross-check that didn't make it into the final note), `drafts/` (earlier standalone LaTeX drafts of the damage and SBS sections, now folded into `BB1GCR_note.tex`). |

A handful of `scripts/calibration/` vs `scripts/flagged_readout/` placements are a
best-effort read of filenames and outputs rather than a line-by-line trace — happy to
be corrected if any script sits in the wrong folder.

## Main results

| Result | Number | Note section | Code |
|---|---|---|---|
| GCR is the best any deterministic unitary achieves at the no-error point | — | §"Why the physical pulse fails" | `scripts/ideal_and_no_unitary_fix/` |
| Heralded (flagged) BB1(GCR) readout, post-selected on ~17% pass rate | infidelity 5.7×10⁻⁴ | §"The flagged readout" | `scripts/flagged_readout/` |
| Variationally compiled BB1(GCR), deterministic, no post-selection | infidelity 1.5×10⁻³ (~5× below bare BB1, before noise) | §"Deterministic readout by variational compilation" | `scripts/variational_compilation/` |
| Once realistic noise (gate duration, decoherence) is included, a single bare GCR beats both heralded and variationally-compiled BB1(GCR) | ~8× | §"Readout under realistic noise: master-equation simulation" | `scripts/mesolve_noise/` |
| Stress test: reading a codeword that idled under decay/dephasing before readout | ranking unchanged | §"Reading a pre-damaged codeword" | `scripts/damage_readout/` |
| Stress test: reading a codeword actively stabilized by noisy SBS rounds | ranking unchanged | §"Reading an actively stabilized codeword" | `scripts/sbs_readout/` |

**Bottom line:** BB1(GCR), whether heralded or deterministically compiled, is not a
good end-of-line readout once physical realizability and noise are accounted for — a
single bare GCR pulse wins. BB1(GCR) remains valuable as proof that the ideal
correction is physically recoverable, and as the tool of choice wherever
post-selection or re-preparation is free, e.g. calibration and known-state
preparation (`scripts/calibration/`).
