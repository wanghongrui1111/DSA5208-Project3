#!/usr/bin/env python3
"""Figures and summary tables for the report.

All directories passed in one call must come from the same machine (speedups
divide by that machine's serial time). CSVs with the same name are merged, so
a cluster's single-node and multi-node job directories can be combined:

  python3 scripts/plot.py --label "my machine" --out figures/local results/<host>
  python3 scripts/plot.py --label Vanda --out figures/vanda \
      results/cluster/cpu_1node results/cluster/cpu_4node results/cluster/gpu

Each directory may contain bench/ and validation/ subdirectories or the CSVs
directly. Requires pandas and matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (light mode) in its fixed slot order. Slots 3-4 sit below
# 3:1 contrast on the surface, so every series also carries a marker shape and
# the numbers are repeated in tables.md.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

KEY = ["impl", "n", "ranks", "threads", "decomp", "comm", "gpu_mpi", "reduce", "cg", "tag"]
INTS = ("n", "ranks", "threads")
# Colour follows the entity: the same CG variant has the same slot everywhere.
CG_SLOT = {"classic": 0, "cgcg": 1, "pipe": 2}
CG_LABEL = {"classic": "classic CG", "cgcg": "Chronopoulos-Gear CG", "pipe": "pipelined CG"}
CONFIGS = [("1d", "blocking"), ("1d", "overlap"), ("2d", "blocking"), ("2d", "overlap")]
LABEL = ""


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 5, "lines.markeredgewidth": 1.0,
    })


def series(ax, x, y, slot, label, **kw):
    """A line in a fixed palette slot, with a marker ringed in the surface colour."""
    return ax.plot(x, y, color=SLOTS[slot], marker=MARKERS[slot], markeredgecolor=SURFACE, label=label, **kw)


def ideal(ax, x, y):
    ax.plot(x, y, color=MUTED, linewidth=0.8, zorder=1, label="ideal")


def rank_axis(ax, ranks, label):
    """log2 axis; a tick label too close to its right neighbour is dropped."""
    ranks = sorted(int(r) for r in ranks)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    labels, last = [], None
    for r in reversed(ranks):
        keep = last is None or np.log2(last / r) >= 0.4
        labels.append(str(r) if keep else "")
        last = r if keep else last
    ax.set_xticklabels(labels[::-1])
    ax.minorticks_off()
    ax.set_xlabel(label)


def header(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=10, fontweight="bold")


def save(fig, out, name):
    fig.tight_layout()
    if LABEL:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.03)
        fig.text(0.995, 0.005, LABEL, ha="right", va="bottom", color=MUTED, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def load(dirs, name):
    frames = []
    for d in dirs:
        for sub in ("", "bench", "validation"):
            p = Path(d) / sub / f"{name}.csv"
            if p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in KEY:
        if c in df:
            df[c] = df[c].astype(int) if c in INTS else df[c].astype(str)
    # A rerun appends rows; keep the latest measurement of each configuration.
    return df.drop_duplicates(subset=[c for c in KEY if c in df], keep="last").reset_index(drop=True)


def serial_time(serial, n, cg="classic"):
    """Serial time per iteration in ms for grid n, or None."""
    if serial is None:
        return None
    s = serial[(serial.n == n) & (serial.cg == cg)]
    return None if s.empty else float(s.time_per_iter_ms.iloc[-1])


def table(title, header_row, rows):
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = [f"## {title}\n", "| " + " | ".join(header_row) + " |", "|" + "---|" * len(header_row)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- validation


def fig_convergence(dirs, out, tables):
    df = load(dirs, "convergence")
    if df is None:
        return
    df = df.sort_values("n")
    h = 1.0 / (df.n.values + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    series(ax, h, df.err_inf.values, 0, "max-norm error")
    series(ax, h, df.err_l2.values, 1, "discrete L2 error")
    ax.plot(h, 1.8 * df.err_inf.values[0] * (h / h[0]) ** 2, color=MUTED, linewidth=0.8, zorder=1, label="slope 2")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("error against the exact solution")
    ax.set_title("Discretisation error under grid refinement")
    ax.legend(loc="lower right")
    save(fig, out, "convergence")
    order = [""] + [f"{np.log(a / b) / np.log(ha / hb):.3f}" for a, b, ha, hb in
                    zip(df.err_inf.values, df.err_inf.values[1:], h, h[1:])]
    tables.append(table("Grid refinement (serial, tol 1e-12)", ["n", "h", "iterations", "max error", "order"],
                        [(r.n, 1 / (r.n + 1), r.iters, r.err_inf, o) for r, o in zip(df.itertuples(), order)]))


def fig_history(dirs, out):
    hist = None
    for d in dirs:
        for sub in ("history", "validation/history"):
            if (Path(d) / sub).is_dir():
                hist = Path(d) / sub
    if hist is None:
        return
    impls = [("serial", "serial"), ("mpi_omp", "MPI+OpenMP, 8 ranks"), ("mpi_cuda", "MPI+CUDA, 2 ranks")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    for ax, cg in zip(axes, ["classic", "cgcg", "pipe"]):
        for slot, (impl, name) in enumerate(impls):
            f = hist / f"{impl}_{cg}.txt"
            if f.exists():
                it, rr = np.loadtxt(f, unpack=True)
                # Serial underneath and wide, so identical parallel curves stay visible on top.
                ax.plot(it, rr, color=SLOTS[slot], label=name, linewidth=3.5 if slot == 0 else 1.0, zorder=1 + slot)
        ax.set_yscale("log")
        ax.set_title(CG_LABEL[cg])
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("recurrence residual ||r|| / ||b||")
    axes[0].legend(loc="lower left")
    header(fig, "Residual histories, n = 511")
    save(fig, out, "residual_history")


# ---------------------------------------------------------------- CPU scaling


def fig_strong(dirs, out, tables):
    df = load(dirs, "strong")
    serial = load(dirs, "serial")
    if df is None:
        return
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(2, len(ns), figsize=(4.5 * len(ns), 6.3), squeeze=False)
    for col, n in enumerate(ns):
        d = df[(df.n == n) & (df.cg == "classic")]
        ranks = np.array(sorted(d.ranks.unique()))
        one = float(d[d.ranks == 1].time_per_iter_ms.iloc[0])
        base = serial_time(serial, n) or one
        ax_t, ax_s = axes[0][col], axes[1][col]
        for slot, (dec, comm) in enumerate(CONFIGS):
            s = d[((d.decomp == dec) & (d.comm == comm)) | (d.ranks == 1)].sort_values("ranks")
            series(ax_t, s.ranks.values, s.time_per_iter_ms.values, slot, f"{dec}, {comm}")
            series(ax_s, s.ranks.values, base / s.time_per_iter_ms.values, slot, f"{dec}, {comm}")
        ideal(ax_t, ranks, one / ranks)
        ideal(ax_s, ranks, ranks.astype(float))
        for ax in (ax_t, ax_s):
            rank_axis(ax, ranks, "MPI ranks (1 thread each)")
        ax_t.set_yscale("log")
        ax_s.set_yscale("log", base=2)
        ax_t.set_title(f"Time per iteration, n = {n}")
        ax_s.set_title(f"Speedup over the serial code, n = {n}")
        ax_t.set_ylabel("ms per CG iteration")
        ax_s.set_ylabel("T_serial / T_parallel")
    axes[0][0].legend(title="decomposition, halo", title_fontsize=8, loc="lower left")
    header(fig, "Strong scaling, classic CG")
    save(fig, out, "strong_scaling")

    # Parallel efficiency of the fastest configuration at each rank count.
    fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 3.3), squeeze=False, sharey=True)
    rows = []
    for col, n in enumerate(ns):
        ax = axes[0][col]
        for cg in ["classic", "cgcg", "pipe"]:
            d = df[(df.n == n) & (df.cg == cg)]
            ts = serial_time(serial, n, cg) or float(d[d.ranks == 1].time_per_iter_ms.iloc[0])
            best = d.loc[d.groupby("ranks").time_per_iter_ms.idxmin()].sort_values("ranks")
            eff = ts / best.time_per_iter_ms.values / best.ranks.values
            series(ax, best.ranks.values, eff, CG_SLOT[cg], CG_LABEL[cg])
            rows += [(n, cg, r.ranks, f"{r.decomp}/{r.comm}", r.time_per_iter_ms, ts / r.time_per_iter_ms, e)
                     for r, e in zip(best.itertuples(), eff)]
        ax.axhline(1.0, color=MUTED, linewidth=0.8, zorder=1)
        rank_axis(ax, df.ranks.unique(), "MPI ranks")
        ax.set_title(f"Parallel efficiency, fastest configuration, n = {n}")
    axes[0][0].set_ylabel("T_serial / (P T_P)")
    axes[0][0].legend(loc="lower left")
    save(fig, out, "strong_efficiency")
    tables.append(table("Strong scaling, fastest configuration per rank count",
                        ["n", "CG", "ranks", "decomp/halo", "ms/iter", "speedup", "efficiency"], rows))


def fig_breakdown(dirs, out):
    df = load(dirs, "strong")
    if df is None:
        return
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 3.3), squeeze=False, sharey=True)
    for col, n in enumerate(ns):
        d = df[(df.n == n) & (df.cg == "classic") & (((df.decomp == "2d") & (df.comm == "blocking")) |
                                                     (df.ranks == 1))].sort_values("ranks")
        halo = (d.t_halo_avg_s / d.time_s).values
        red = (d.t_reduce_avg_s / d.time_s).values
        x = np.arange(len(d))
        ax = axes[0][col]
        bottom = np.zeros(len(d))
        for slot, (part, name) in enumerate([(1.0 - halo - red, "computation"), (halo, "halo exchange"),
                                             (red, "global reductions")]):
            ax.bar(x, 100 * part, bottom=100 * bottom, width=0.6, color=SLOTS[slot], label=name,
                   edgecolor=SURFACE, linewidth=1.0)
            bottom += part
        ax.set_xticks(x)
        ax.set_xticklabels(d.ranks.values)
        ax.set_xlabel("MPI ranks")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"n = {n}")
    axes[0][0].set_ylabel("% of solve time (average over ranks)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    header(fig, "Where the time goes: classic CG, 2d decomposition, blocking halo")
    save(fig, out, "time_breakdown")


def fig_variants(dirs, out):
    df = load(dirs, "strong")
    if df is None:
        return
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(2, len(ns), figsize=(4.5 * len(ns), 6.1), squeeze=False)
    for col, n in enumerate(ns):
        d = df[(df.n == n) & (((df.decomp == "2d") & (df.comm == "overlap")) | (df.ranks == 1))]
        for cg in ["classic", "cgcg", "pipe"]:
            s = d[d.cg == cg].sort_values("ranks")
            its = s.time_s.values / s.time_per_iter_ms.values * 1e3  # iterations per solve
            series(axes[0][col], s.ranks.values, s.time_per_iter_ms.values, CG_SLOT[cg], CG_LABEL[cg])
            series(axes[1][col], s.ranks.values, 1e3 * s.t_reduce_max_s.values / its, CG_SLOT[cg], CG_LABEL[cg])
        for ax in axes[:, col]:
            rank_axis(ax, d.ranks.unique(), "MPI ranks")
            ax.set_yscale("log")
        axes[0][col].set_title(f"Time per iteration, n = {n}")
        axes[1][col].set_title(f"Time inside reductions per iteration, n = {n}")
    axes[0][0].set_ylabel("ms per CG iteration")
    axes[1][0].set_ylabel("ms per iteration (slowest rank)")
    axes[0][0].legend(loc="lower left")
    header(fig, "CG variants: 2d decomposition, overlapped halo")
    save(fig, out, "cg_variants")


def fig_halo(dirs, out, tables):
    """Classic CG with every halo strategy: neighbour exchange vs one-sided vs full copy."""
    df = load(dirs, "strong")
    if df is None or "rma" not in set(df.comm):
        return
    halos = [("1d", "blocking"), ("1d", "allgather"), ("2d", "blocking"), ("2d", "overlap"), ("2d", "rma")]
    names = {"blocking": "Sendrecv", "overlap": "Isend/Irecv + overlap", "rma": "MPI_Put + fence",
             "allgather": "Allgatherv (full copy)"}
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(2, len(ns), figsize=(4.5 * len(ns), 6.1), squeeze=False)
    rows = []
    for col, n in enumerate(ns):
        d = df[(df.n == n) & (df.cg == "classic")]
        for slot, (dec, comm) in enumerate(halos):
            s = d[(d.decomp == dec) & (d.comm == comm) & (d.ranks > 1)].sort_values("ranks")
            its = s.time_s.values / s.time_per_iter_ms.values * 1e3
            label = f"{dec}, {names[comm]}"
            series(axes[0][col], s.ranks.values, s.time_per_iter_ms.values, slot, label)
            series(axes[1][col], s.ranks.values, 1e3 * s.t_halo_max_s.values / its, slot, label)
            rows += [(n, r.ranks, dec, comm, r.time_per_iter_ms, 1e3 * r.t_halo_max_s / i)
                     for r, i in zip(s.itertuples(), its)]
        for ax in axes[:, col]:
            rank_axis(ax, d[d.ranks > 1].ranks.unique(), "MPI ranks (1 thread each)")
            ax.set_yscale("log")
        axes[0][col].set_title(f"Time per iteration, n = {n}")
        axes[1][col].set_title(f"Halo time per iteration (slowest rank), n = {n}")
    axes[0][0].set_ylabel("ms per CG iteration")
    axes[1][0].set_ylabel("ms per iteration")
    axes[1][0].legend(loc="upper left", fontsize=7.5)
    header(fig, "Halo strategies, classic CG")
    save(fig, out, "halo_strategies")
    tables.append(table("Halo strategies, classic CG", ["n", "ranks", "decomp", "halo", "ms/iter", "halo ms/iter"],
                        rows))


def fig_repro(dirs, out, tables):
    """What the reproducible (fixed-order) reductions cost."""
    df = load(dirs, "repro")
    if df is not None:
        ns = sorted(df.n.unique())
        fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 3.3), squeeze=False)
        rows = []
        for col, n in enumerate(ns):
            ax = axes[0][col]
            d = df[df.n == n]
            for slot, red in enumerate(["fast", "repro"]):
                s = d[d.reduce == red].sort_values("ranks")
                series(ax, s.ranks.values, s.time_per_iter_ms.values, slot,
                       "MPI_Allreduce + OpenMP reduction" if red == "fast" else "fixed-order sums (repro)")
            both = d.pivot_table(index="ranks", columns="reduce", values="time_per_iter_ms")
            rows += [(n, r, v.get("fast"), v.get("repro"), 100 * (v.get("repro") / v.get("fast") - 1))
                     for r, v in both.iterrows()]
            rank_axis(ax, d.ranks.unique(), "MPI ranks (1 thread each)")
            ax.set_yscale("log")
            ax.set_title(f"n = {n}")
        axes[0][0].set_ylabel("ms per CG iteration")
        axes[0][0].legend(loc="lower left")
        header(fig, "Cost of reproducible reductions, classic CG")
        save(fig, out, "repro_cost")
        tables.append(table("Cost of reproducible reductions (ms per iteration)",
                            ["n", "ranks", "fast", "repro", "overhead %"], rows))
    val = load(dirs, "reproducibility")
    if val is not None:
        rows = [(impl, comm, red, len(g), g.relres_bits.nunique())
                for (impl, comm, red), g in val.groupby(["impl", "comm", "reduce"])]
        tables.append(table("Reproducibility: distinct final residuals over repeated runs with 1/2/4 threads",
                            ["implementation", "halo", "reduction", "runs", "distinct bit patterns"], rows))


def fig_placement(dirs, out, tables):
    df = load(dirs, "placement")
    if df is None:
        return
    ns = sorted(df.n.unique())
    names = {"map=slot": "packed: fill socket 0 first", "map=socket": "spread: alternate sockets"}
    fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 3.3), squeeze=False)
    rows = []
    for col, n in enumerate(ns):
        ax = axes[0][col]
        for slot, tag in enumerate(names):
            s = df[(df.n == n) & (df.tag == tag)].sort_values("ranks")
            series(ax, s.ranks.values, s.time_per_iter_ms.values, slot, names[tag])
        both = df[df.n == n].pivot_table(index="ranks", columns="tag", values="time_per_iter_ms")
        rows += [(n, r, v.get("map=slot"), v.get("map=socket"), v.get("map=slot") / v.get("map=socket"))
                 for r, v in both.iterrows()]
        rank_axis(ax, df.ranks.unique(), "MPI ranks (1 thread each)")
        ax.set_yscale("log")
        ax.set_title(f"n = {n}")
    axes[0][0].set_ylabel("ms per CG iteration")
    axes[0][0].legend(loc="lower left")
    header(fig, "Rank placement on a two-socket node, classic CG")
    save(fig, out, "placement")
    tables.append(table("Rank placement (ms per iteration)", ["n", "ranks", "packed", "spread", "packed/spread"],
                        rows))


def fig_hybrid(dirs, out, tables):
    df = load(dirs, "hybrid")
    if df is None:
        return
    df = df.assign(cores=df.ranks * df.threads)
    # One panel per (core count, n): a multi-node job adds its own core count.
    panels = sorted({(int(c), int(n)) for c, n in zip(df.cores, df.n)})
    nsplit = df.groupby("cores").threads.nunique().max()
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(6.0, 0.75 * nsplit + 1.5), 3.2 * len(panels)),
                             squeeze=False)
    variants = [("classic", "overlap"), ("classic", "thread"), ("pipe", "overlap"), ("pipe", "thread")]
    names = {"overlap": "Isend/Irecv overlap", "thread": "master thread communicates"}
    rows = []
    for row, (cores, n) in enumerate(panels):
        ax = axes[row][0]
        d = df[(df.n == n) & (df.cores == cores)]
        splits = sorted(d.threads.unique())
        x = np.arange(len(splits))
        for k, (cg, comm) in enumerate(variants):
            s = d[(d.cg == cg) & (d.comm == comm)].set_index("threads").reindex(splits)
            if s.time_per_iter_ms.isna().all():
                continue
            ax.bar(x + (k - 1.5) * 0.2, s.time_per_iter_ms.values, width=0.19, color=SLOTS[k],
                   label=f"{CG_LABEL[cg]}, {names[comm]}", edgecolor=SURFACE, linewidth=1.0)
            rows += [(cores, n, cg, comm, f"{int(r.ranks)} x {t}", r.time_per_iter_ms) for t, r in s.iterrows()
                     if not np.isnan(r.time_per_iter_ms)]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{cores // t}x{t}" for t in splits])
        ax.set_xlabel("MPI ranks x OpenMP threads")
        ax.grid(axis="x", visible=False)
        nodes = int(d.nodes.max()) if "nodes" in d else 1
        ax.set_title(f"{cores} cores ({nodes} node{'s' if nodes > 1 else ''}), n = {n}")
        ax.set_ylabel("ms per CG iteration")
    axes[0][0].legend(loc="upper left", ncol=2)
    header(fig, "Hybrid MPI+OpenMP: ranks x threads on all cores")
    save(fig, out, "hybrid")
    tables.append(table("Hybrid MPI+OpenMP, all cores", ["cores", "n", "CG", "halo", "ranks x threads", "ms/iter"],
                        rows))


def fig_weak(dirs, out, name="weak", unit="process", xlabel="MPI ranks (1 thread each)"):
    df = load(dirs, name)
    if df is None:
        return
    # Against one process, weak scaling inside a node mostly measures how the
    # node's memory bandwidth is shared. With several nodes, a second panel
    # compares against one full node, which isolates the cost of going multi-node.
    full = int(df[df.nodes == 1].ranks.max()) if "nodes" in df and df.nodes.max() > 1 else None
    bases = [(1, f"relative to 1 {unit}")] + ([(full, f"relative to 1 full node ({full} {unit}es)"
                                                 if unit == "process" else f"relative to 1 full node")]
                                                if full else [])
    fig, axes = plt.subplots(1, len(bases), figsize=(5.0 * len(bases), 3.4), squeeze=False)
    variants = [("classic", "blocking"), ("classic", "overlap"), ("pipe", "blocking"), ("pipe", "overlap")]
    local = str(df.tag.iloc[0]).replace("local=", "")
    for ax, (base_ranks, title) in zip(axes[0], bases):
        for slot, (cg, comm) in enumerate(variants):
            s = df[(df.comm == comm) & (df.cg == cg) & (df.ranks >= base_ranks)].sort_values("ranks")
            if s.empty:
                continue
            base = float(s[s.ranks == base_ranks].time_per_iter_ms.iloc[0])
            series(ax, s.ranks.values, base / s.time_per_iter_ms.values, slot, f"{CG_LABEL[cg]}, {comm} halo")
        ax.axhline(1.0, color=MUTED, linewidth=0.8, zorder=1)
        rank_axis(ax, df[df.ranks >= base_ranks].ranks.unique(), xlabel)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
    axes[0][0].set_ylabel(f"T_iter(base) / T_iter(P)")
    axes[0][0].legend(loc="lower left")
    header(fig, f"Weak scaling, {local}x{local} points per {unit}")
    save(fig, out, name)


# ---------------------------------------------------------------- GPU


def fig_gpu(dirs, out, tables):
    gpu = load(dirs, "gpu")
    strong = load(dirs, "strong")
    hybrid = load(dirs, "hybrid")
    serial = load(dirs, "serial")
    if gpu is None:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for cg in ["classic", "cgcg", "pipe"]:
        s = gpu[gpu.cg == cg].sort_values("n")
        series(ax, s.n.values, s.mlups.values, CG_SLOT[cg], CG_LABEL[cg])
    sizes = sorted(gpu.n.unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(v) for v in sizes])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("grid points per dimension n")
    ax.set_ylabel("million grid-point updates / s")
    ax.set_title(f"One GPU ({gpu.device.iloc[0]})")
    ax.legend(loc="lower right")
    save(fig, out, "gpu_sizes")
    tables.append(table("One GPU", ["n", "CG", "ms/iter", "MLUPS"],
                        [(r.n, r.cg, r.time_per_iter_ms, r.mlups) for r in gpu.sort_values(["n", "cg"]).itertuples()]))

    # Serial vs a full CPU node vs one GPU, at the sizes all three ran. The
    # node time is the fastest all-core run, pure MPI (strong) or hybrid.
    node_runs = [d for d in (strong, hybrid) if d is not None]
    if not node_runs or serial is None:
        return
    node_runs = pd.concat(node_runs, ignore_index=True)
    node_runs = node_runs[node_runs.cg == "classic"]
    node_runs = node_runs[node_runs.ranks * node_runs.threads == (node_runs.ranks * node_runs.threads).max()]
    cores = int((node_runs.ranks * node_runs.threads).max())
    rows, groups = [], []
    for n in sorted(set(gpu.n) & set(node_runs.n) & set(serial.n)):
        st = serial_time(serial, n)
        node = float(node_runs[node_runs.n == n].time_per_iter_ms.min())
        g = float(gpu[(gpu.n == n) & (gpu.cg == "classic")].time_per_iter_ms.iloc[0])
        groups.append((n, [st, node, g]))
        rows.append((n, st, node, g, st / node, st / g, node / g))
    if not groups:
        return
    names = ["serial\n(1 core)", f"MPI\n({cores} cores)", "MPI+CUDA\n(1 GPU)"]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.3), squeeze=False)
    for ax, (n, vals) in zip(axes[0], groups):
        x = np.arange(3)
        ax.bar(x, vals, width=0.55, color=SLOTS[:3], edgecolor=SURFACE, linewidth=1.0)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3g} ms", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        color=INK2, fontsize=7.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"n = {n}")
    axes[0][0].set_ylabel("ms per CG iteration (classic)")
    header(fig, "Serial vs full CPU node vs one GPU")
    save(fig, out, "cpu_vs_gpu")
    tables.append(table("Serial vs CPU node vs GPU, classic CG (ms per iteration)",
                        ["n", "serial", f"{cores} cores", "1 GPU", "node speedup", "GPU speedup", "GPU vs node"],
                        rows))


def fig_gpu_strong(dirs, out, tables):
    df = load(dirs, "gpu_strong")
    if df is None or df.ranks.max() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows = []
    for ax, cg in zip(axes, ["classic", "pipe"]):
        d = df[df.cg == cg]
        for slot, (comm, gm) in enumerate([("blocking", "staged"), ("blocking", "aware"), ("overlap", "staged"),
                                           ("overlap", "aware")]):
            s = d[((d.comm == comm) & (d.gpu_mpi == gm)) | (d.ranks == 1)].sort_values("ranks")
            series(ax, s.ranks.values, s.time_per_iter_ms.values, slot, f"{comm} halo, {gm} MPI")
            rows += [(cg, r.ranks, comm, gm, r.time_per_iter_ms) for r in s.itertuples() if r.ranks > 1]
        ranks = np.array(sorted(d.ranks.unique()))
        ideal(ax, ranks, float(d[d.ranks == 1].time_per_iter_ms.iloc[0]) / ranks)
        rank_axis(ax, ranks, "GPUs (one MPI rank each)")
        ax.set_yscale("log")
        ax.set_title(f"{CG_LABEL[cg]}, n = {int(d.n.iloc[0])}")
    axes[0].set_ylabel("ms per CG iteration")
    axes[0].legend(loc="lower left")
    header(fig, "Strong scaling over GPUs")
    save(fig, out, "gpu_strong")
    tables.append(table("Strong scaling over GPUs (ms per iteration)", ["CG", "GPUs", "halo", "MPI", "ms/iter"],
                        rows))


def main():
    global LABEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="machine name printed in the corner of every figure")
    a = ap.parse_args()
    LABEL = a.label
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    tables = [f"# Result tables ({a.label})\n"]
    fig_convergence(a.dirs, out, tables)
    fig_history(a.dirs, out)
    fig_strong(a.dirs, out, tables)
    fig_breakdown(a.dirs, out)
    fig_variants(a.dirs, out)
    fig_halo(a.dirs, out, tables)
    fig_repro(a.dirs, out, tables)
    fig_placement(a.dirs, out, tables)
    fig_hybrid(a.dirs, out, tables)
    fig_weak(a.dirs, out)
    fig_gpu(a.dirs, out, tables)
    fig_gpu_strong(a.dirs, out, tables)
    fig_weak(a.dirs, out, "gpu_weak", "GPU", "GPUs (one MPI rank each)")
    (out / "tables.md").write_text("\n".join(tables))
    print("wrote", out / "tables.md")


if __name__ == "__main__":
    main()
