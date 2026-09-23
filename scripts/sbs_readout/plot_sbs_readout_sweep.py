"""SBS-stabilization + readout sweep figures for the note (fig:sbs-readout).

Reads Paper_Data/sbs_authentic_sweep.npz, which sweeps the number of SBS
rounds m in {0,2,4,6,8,10} and reports codeword-L0 readout error P(e) and
post-read oscillator infidelity 1-F under every combination of an SBS-phase
noise condition (sbscond) and a readout-phase noise condition (readcond).

Only the SIX unitary correction protocols are plotted (herald-RUS is
excluded: its momentum-filter is not shift-covariant to the binomial
lattice used here, so it reads at chance ~0.5 on this codeword and is
handled separately in the text).

Produces two figures, matching the aesthetic of plot_damage_readout.py:

  Figure 1 (Paper_Figures/sbs_readout_sweep.pdf/.png):
    2x5 grid, rows = metric (P(e) top, 1-F bottom), columns = the 5 SBS-noise
    conditions, readout FIXED to 'noiseless' -- isolates SBS-noise damage.

  Figure 2 (Paper_Figures/sbs_readout_conds.pdf/.png):
    2x6 grid, rows = metric, columns = the 6 readout conditions, SBS
    condition FIXED to 'complete' (realistic, all channels active).

Both figures use a LOG y-axis (per request, to make the small protocol-to-
protocol differences on the high binomial-SBS floor as explicit as possible).
The values live in narrow, sub-decade bands (P(e)~0.078-0.12, 1-F~0.03-0.08),
so the single decade tick at 0.1 is supplemented with EXPLICIT per-row fixed
ticks; axes are shared per row (sharey by row, sharex across all panels), and
reuse the canonical SCOL/SLAB color/label mapping from plot_damage_readout.py.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

_rob = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Roboto-Regular.ttf")
if os.path.exists(_rob): fm.fontManager.addfont(_rob)
plt.rcParams.update({'font.family': 'Roboto', 'mathtext.fontset': 'cm', 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.linewidth': 2.2,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.fontsize': 11,
    'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.major.size': 4, 'ytick.major.size': 4})

d = np.load("Paper_Data/sbs_authentic_sweep.npz")
m = np.asarray(d['m_list'])

# canonical protocol set (herald-RUS excluded -- see module docstring)
schemes = ['GCR', 'BB1(GCR)-QITE', 'BB1', 'GCR-BB1', 'GCR-BB1-blockA', 'GCR-BB1-blockB']
SCOL = {'GCR': 'firebrick', 'BB1(GCR)-QITE': 'darkviolet', 'BB1': 'royalblue', 'GCR-BB1': 'forestgreen',
        'GCR-BB1-blockA': 'mediumseagreen', 'GCR-BB1-blockB': 'darkcyan'}
SLAB = {'GCR': 'single GCR', 'BB1(GCR)-QITE': 'BB1(GCR) det.', 'BB1': 'bare BB1', 'GCR-BB1': 'GCR-BB1 (1 pre-corr.)',
        'GCR-BB1-blockA': 'GCR-BB1 (block A)', 'GCR-BB1-blockB': 'GCR-BB1 (block B)'}
# draw order: most-overlapped/lowest schemes first so the busy ones land on top
ZORD = {'BB1': 1, 'GCR-BB1': 2, 'GCR': 3, 'GCR-BB1-blockA': 4, 'GCR-BB1-blockB': 5, 'BB1(GCR)-QITE': 6}

ROWS = [('err', r'readout error  $P(e)$'),
        ('infid', r'post-read infidelity  $1-F$')]


def val(proto, sbscond, readcond, metric):
    return np.asarray(d[f'{proto}|{sbscond}|{readcond}|L0|{metric}'])


def draw(ax_row, metric, cond_list, fixed, mode):
    """mode='sbs' -> cond_list are sbsconds, readout fixed to `fixed`.
       mode='read' -> cond_list are readconds, sbscond fixed to `fixed`."""
    for j, cond in enumerate(cond_list):
        a = ax_row[j]
        for sc in schemes:
            if mode == 'sbs':
                y = val(sc, cond, fixed, metric)
            else:
                y = val(sc, fixed, cond, metric)
            a.plot(m, y, '-', color=SCOL[sc], lw=1.8, marker='o', ms=3.5, zorder=ZORD[sc])
        a.set_xlim(m[0], m[-1]); a.set_xticks(m)
        a.set_yscale('log')
        a.grid(alpha=0.35, which='both')


# explicit sub-decade ticks per metric (the only decade tick in range is 0.1)
YTICKS = {'err': [0.08, 0.09, 0.10, 0.11, 0.12],
          'infid': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]}


def set_logticks(ax_row, metric):
    """On a shared-per-row log axis, replace the sparse decade ticks with an
    explicit fixed set so the narrow band is actually legible."""
    ticks = YTICKS[metric]
    a0 = ax_row[0]
    a0.yaxis.set_major_locator(FixedLocator(ticks))
    a0.yaxis.set_major_formatter(FixedFormatter([f'{t:.2f}' for t in ticks]))
    a0.yaxis.set_minor_locator(NullLocator())


def make_legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=SCOL[sc], lw=1.8, marker='o', ms=3.5, label=SLAB[sc]) for sc in schemes]


# ---------------------------------------------------------------------------
# Figure 1: SBS-noise sweep, readout fixed to 'noiseless'
# ---------------------------------------------------------------------------
sbs_conds = ['complete', 'tr_decay', 'tr_deph', 'osc_decay', 'osc_deph']
sbs_titles = ['all channels', 'transmon decay', 'transmon deph.', 'oscillator decay', 'oscillator deph.']

fig, ax = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True, sharey='row')
for i, (metric, ylab) in enumerate(ROWS):
    draw(ax[i], metric, sbs_conds, 'noiseless', 'sbs')
    set_logticks(ax[i], metric)
    ax[i, 0].set_ylabel(ylab, fontsize=13)
for j, title in enumerate(sbs_titles):
    ax[0, j].set_title(title, fontsize=13)
for j in range(5):
    ax[1, j].set_xlabel(r'SBS rounds  $m$')

handles = make_legend_handles()
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig("Paper_Figures/sbs_readout_sweep.pdf", bbox_inches="tight", transparent=True)
fig.savefig("Paper_Figures/sbs_readout_sweep.png", dpi=150, bbox_inches="tight")
print("wrote Paper_Figures/sbs_readout_sweep.pdf")
print("Figure 1 y-limits:", [ax[i, 0].get_ylim() for i in range(2)])
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: readout-condition sweep, SBS condition fixed to 'complete'
# ---------------------------------------------------------------------------
read_conds = ['noiseless', 'complete', 'tr_decay', 'tr_deph', 'osc_decay', 'osc_deph']
read_titles = ['noiseless read', 'all-ch. read', 'tr. decay read', 'tr. deph. read', 'osc. decay read', 'osc. deph. read']

fig2, ax2 = plt.subplots(2, 6, figsize=(17, 6.2), sharex=True, sharey='row')
for i, (metric, ylab) in enumerate(ROWS):
    draw(ax2[i], metric, read_conds, 'complete', 'read')
    set_logticks(ax2[i], metric)
    ax2[i, 0].set_ylabel(ylab, fontsize=13)
for j, title in enumerate(read_titles):
    ax2[0, j].set_title(title, fontsize=13)
for j in range(6):
    ax2[1, j].set_xlabel(r'SBS rounds  $m$')

handles2 = make_legend_handles()
fig2.legend(handles=handles2, loc='lower center', ncol=6, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
fig2.tight_layout(rect=[0, 0.08, 1, 1])
fig2.savefig("Paper_Figures/sbs_readout_conds.pdf", bbox_inches="tight", transparent=True)
fig2.savefig("Paper_Figures/sbs_readout_conds.png", dpi=150, bbox_inches="tight")
print("wrote Paper_Figures/sbs_readout_conds.pdf")
print("Figure 2 y-limits:", [ax2[i, 0].get_ylim() for i in range(2)])
plt.close(fig2)
