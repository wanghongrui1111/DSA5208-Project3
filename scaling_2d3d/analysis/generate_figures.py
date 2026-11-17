"""Rebuild the combined Vanda scaling figures (strong and weak) from the accompanying CSV summaries.
Run after analysis/aggregate.py:  python3 analysis/generate_figures.py   (needs matplotlib)

Strong scaling: analysis/e2_points.csv -> 02_single_node, 03_multi_node, 04_components
Weak scaling:   analysis/e8_points.csv -> 05_weak_scaling, 06_weak_components
The palette follows the supplied project figures (Figures 3-11 of p3.docx): blue, vermilion, green, amber
on an off-white background, with a thin grey ideal line.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
DATA = ROOT / 'analysis'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']      # palette of the supplied project figures
MK = ['o', 's', '^', 'D']
BG = '#fcfcfb'
IDEAL = '#808080'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.labelsize': 9, 'axes.labelcolor': '#444444',
                     'legend.fontsize': 7.4, 'legend.frameon': False, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'xtick.color': '#444444', 'ytick.color': '#444444', 'axes.edgecolor': '#bbbbbb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e4e4', 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'lines.linewidth': 1.6, 'lines.markersize': 5, 'lines.markeredgecolor': 'white',
                     'lines.markeredgewidth': .7, 'figure.facecolor': BG, 'axes.facecolor': BG,
                     'savefig.facecolor': BG, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def load(name):
    with open(DATA / name, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = int(v)
            except ValueError:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def sel(rows, **kw):
    return sorted((r for r in rows if all(r[k] == v for k, v in kw.items())), key=lambda r: r['ranks'])


def col(rows, k):
    return [r[k] for r in rows]


def save(fig, name):
    fig.tight_layout(pad=.8)
    for ext in ['pdf', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=230, bbox_inches='tight')
    plt.close(fig)


def logx(ax, x, label='MPI ranks (1 thread each)', hide=()):
    x = sorted(set(int(v) for v in x)); ax.set_xscale('log', base=2); ax.set_xticks(x)
    ax.set_xticklabels(['' if v in hide else str(v) for v in x]); ax.minorticks_off(); ax.set_xlabel(label)


def line(ax, x, y, i, label, **kw):
    return ax.plot(x, y, color=C[i % 4], marker=MK[i % 4], label=label, **kw)


def ideal_h(ax):
    ax.axhline(1, color=IDEAL, lw=.9, zorder=1)


# ---------------------------------------------------------------- strong scaling (E2)
E = load('e2_points.csv')

fig, axs = plt.subplots(2, 2, figsize=(6.85, 4.75))
for c, (dim, sizes, dc) in enumerate([(2, [1024, 4096], '2d'), (3, [100, 256], '3d')]):
    i = 0
    for n in sizes:
        for halo, ls in [('block', '-'), ('overlap', '--')]:
            d = sel(E, experiment='E2a', dim=dim, n=n, decomp=dc, halo=halo)
            lab = f'n={n}, {"blocking" if halo == "block" else "overlap"}'
            line(axs[0, c], col(d, 'ranks'), col(d, 'ms_per_iter'), i, lab, linestyle=ls)
            line(axs[1, c], col(d, 'ranks'), col(d, 'efficiency'), i, lab, linestyle=ls)
            i += 1
    axs[0, c].set_yscale('log'); axs[0, c].set_title(f'({chr(97 + c)}) {dim}D, one Vanda node')
    axs[0, c].set_ylabel('ms per CG iteration'); axs[1, c].set_ylabel('Parallel efficiency')
    ideal_h(axs[1, c])
    for row in range(2):
        logx(axs[row, c], [1, 2, 4, 8, 16, 32, 64, 72], hide=(64,))
    axs[0, c].legend(loc='best')
save(fig, '02_single_node')

fig, axs = plt.subplots(2, 2, figsize=(6.85, 4.85))
multi = [r for r in E if r['experiment'] == 'E2b']
styles = [(2, '2d', 'block', 0, '2D blocks'), (2, '1d', 'block', 1, '2D strips'),
          (3, '3d', 'block', 2, '3D blocking'), (3, '3d', 'overlap', 3, '3D overlap')]
for dim, dc, halo, i, label in styles:
    d = sel(multi, dim=dim, decomp=dc, halo=halo)
    ax = axs[0, 0 if dim == 2 else 1]; x = col(d, 'ranks')
    line(ax, x, col(d, 'ms_per_iter'), i, label)
    ax.fill_between(x, col(d, 'p25_ms'), col(d, 'p75_ms'), color=C[i], alpha=.15, lw=0)
    for a, metric in [(axs[1, 0], 'speedup'), (axs[1, 1], 'efficiency')]:
        line(a, x, col(d, metric), i, label)
    wide = [r for r in d if str(r['wide']).lower() == 'true']
    for a, metric in [(ax, 'ms_per_iter'), (axs[1, 0], 'speedup')]:
        a.scatter(col(wide, 'ranks'), col(wide, metric), s=70, facecolors='none', edgecolors=C[i], lw=1, zorder=5)
for ax in axs.flat:
    logx(ax, [32, 64, 128, 256])
for j in range(2):
    axs[0, j].set_yscale('log'); axs[0, j].set_ylabel('ms per CG iteration'); axs[0, j].legend()
axs[0, 0].set_title('(a) 2D, n=4096'); axs[0, 1].set_title('(b) 3D, n=256')
axs[1, 0].set_yscale('log'); axs[1, 0].plot([32, 256], [32, 256], color=IDEAL, lw=.9, label='ideal')
axs[1, 0].set_ylabel('Speedup over E1 serial'); axs[1, 1].set_ylabel('Parallel efficiency')
axs[1, 0].legend(fontsize=6.8); axs[1, 1].set_ylim(0, 1.04)
save(fig, '03_multi_node')

fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.7))
for ax, (dim, n, dc) in zip(axs, [(2, 4096, '2d'), (3, 256, '3d')]):
    d = sel(multi, dim=dim, n=n, decomp=dc, halo='block')
    for i, (key, lbl) in enumerate([('compute_ms', 'compute'), ('halo_ms', 'halo'), ('reduce_ms', 'reduction + waiting'), ('ms_per_iter', 'total')]):
        line(ax, col(d, 'ranks'), col(d, key), i, lbl, linestyle='--' if key == 'ms_per_iter' else '-')
    logx(ax, col(d, 'ranks')); ax.set(yscale='log', ylabel='ms per CG iteration', title=f'{dim}D, n={n}'); ax.legend(fontsize=7)
save(fig, '04_components')

# ---------------------------------------------------------------- weak scaling (E8)
W = load('e8_points.csv')
CONF = [('block', 'classic', 'blocking, classic'), ('overlap', 'classic', 'overlap, classic'),
        ('block', 'pipelined', 'blocking, pipelined'), ('overlap', 'pipelined', 'overlap, pipelined')]
XT = [1, 2, 4, 8, 16, 32, 64, 72, 128, 256]

fig, axs = plt.subplots(2, 2, figsize=(6.85, 5.0))
for c, (dim, dc, loc) in enumerate([(2, '2d', '1024²'), (3, '3d', '100³')]):
    for i, (halo, cg, lab) in enumerate(CONF):
        if dim == 3 and cg != 'classic':
            continue
        s = sel(W, experiment='E8s', dim=dim, decomp=dc, halo=halo, cg=cg)
        m = sel(W, experiment='E8m', dim=dim, decomp=dc, halo=halo, cg=cg)
        for d, ls, alpha, tag in [(s, '-', 1, 'one node'), (m, '--', .85, '8×32')]:
            x = col(d, 'ranks')
            line(axs[0, c], x, col(d, 'ms_per_iter'), i, f'{lab} ({tag})', linestyle=ls, alpha=alpha)
            if tag == 'one node':
                axs[0, c].fill_between(x, col(d, 'p25_ms'), col(d, 'p75_ms'), color=C[i], alpha=.15, lw=0)
            t0 = d[0]['ms_per_iter']          # E8s: 1 rank; E8m: 32 ranks (first node)
            line(axs[1, c], x, [t0 / t for t in col(d, 'ms_per_iter')], i, f'{lab} ({tag})', linestyle=ls, alpha=alpha)
        wide = [r for r in s if r['within10'] < 2 / 3]
        axs[0, c].scatter(col(wide, 'ranks'), col(wide, 'ms_per_iter'), s=70, facecolors='none', edgecolors=C[i], lw=1, zorder=5)
    axs[0, c].set_title(f'({chr(97 + c)}) {dim}D, {loc} points per rank')
    axs[0, c].set_yscale('log'); axs[0, c].set_ylabel('ms per CG iteration')
    axs[0, c].set_yticks([3, 5, 10, 20]); axs[0, c].set_yticklabels(['3', '5', '10', '20']); axs[0, c].minorticks_off()
    axs[1, c].set_title(f'({chr(99 + c)}) {dim}D weak efficiency')
    axs[1, c].set_ylabel('T(P$_0$) / T(P)'); axs[1, c].set_ylim(0, 1.1); ideal_h(axs[1, c])
    for row in range(2):
        logx(axs[row, c], XT, hide=(64,))
from matplotlib.lines import Line2D
hd = [Line2D([], [], color=C[i], marker=MK[i], label=lab) for i, (_, _, lab) in enumerate(CONF)]
hd += [Line2D([], [], color='#555555', ls='-', label='one 72-core node, P$_0$=1 (3 jobs)'),
       Line2D([], [], color='#555555', ls='--', label='32 ranks/node, 1–8 nodes, P$_0$=32 (1 job)')]
fig.tight_layout(pad=.8, rect=(0, .08, 1, 1))
fig.legend(handles=hd, loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(.5, 0))
for ext in ['pdf', 'png']:
    fig.savefig(FIG / f'05_weak_scaling.{ext}', dpi=230, bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.9))
for ax, (dim, dc) in zip(axs, [(2, '2d'), (3, '3d')]):
    for exp, ls in [('E8s', '-'), ('E8m', '--')]:
        d = sel(W, experiment=exp, dim=dim, decomp=dc, halo='block', cg='classic')
        for i, (key, lbl) in enumerate([('ms_per_iter', 'total'), ('compute_ms', 'compute'), ('halo_ms', 'halo'), ('reduce_ms', 'reduction + waiting')]):
            line(ax, col(d, 'ranks'), col(d, key), i, None, linestyle=ls,
                 lw=3.2 if key == 'ms_per_iter' else 1.4, alpha=.45 if key == 'ms_per_iter' else 1,
                 zorder=2 if key == 'ms_per_iter' else 3)
    logx(ax, XT, hide=(64,)); ax.set(yscale='log', ylabel='ms per CG iteration', title=f'{dim}D, blocking, classic')
    ax.set_ylim(1e-2, 40)
    ax.legend(handles=[Line2D([], [], color=C[i], marker=MK[i], lw=3.2 if i == 0 else 1.4, alpha=.45 if i == 0 else 1, label=l)
                       for i, l in enumerate(['total', 'compute', 'halo', 'reduction + waiting'])], fontsize=6.8, loc='lower right')
fig.text(.99, -.05, 'Solid: one node (1–72 ranks).  Dashed: 32 ranks per node (32–256 ranks).\n'
         'Counters are separately maximised; 1-rank halo/reduction (<0.01 ms) lie off-axis.', ha='right', fontsize=6.8, color='#777777')
save(fig, '06_weak_components')

print('Rebuilt strong-scaling (02-04) and weak-scaling (05-06) Vanda figures.')
