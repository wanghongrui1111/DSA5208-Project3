#!/usr/bin/env python3
"""aggregate.py -- turn the raw per-job CSVs in results/ into the published scaling points.

  python3 analysis/aggregate.py [RESULTS_DIR]     (standard library only; default RESULTS_DIR = results/)

Writes analysis/e2_points.csv (strong scaling, E1 serial baseline) and analysis/e8_points.csv (weak scaling).
Publication rule used in the report:
  * every CSV row is one run (the median-time repeat of that run);
  * a point is the MEDIAN over all rows of all jobs and passes that contain it, with the 25th-75th percentiles;
  * a point needs >= 2 independent PBS jobs (exception: the multi-node weak series, one job, flagged single_job=True);
  * iteration protocol: single-node runs use clamp(100*4096^2/N, 100, 1000); multi-node runs 1000 iterations, >= 5 repeats;
  * 'wide' = fewer than 2/3 of the rows lie within +-10 % of the median (a description, never a rejection);
  * E1 serial: two jobs, accepted when they differ by <= 10 %; their median is the speedup baseline.
Job e8m_r1 is excluded (about 1.7x slower at every point, including the one-node point; see the report).
"""
import csv, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
EXCLUDE = {"e8m_r1"}


def rows(job):
    p = os.path.join(RES, job, ("e1" if job.startswith("e1") else job) + ".csv")   # E1 jobs always write e1.csv
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def ms(r, k):
    return 1e3 * float(r[k]) / max(1, int(r["iters"]))


def quant(v, p):
    v = sorted(v); i = (len(v) - 1) * p; lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def auto_iters(dim, n):
    return int(round(min(1000.0, max(100.0, 100.0 * 4096.0 ** 2 / float(n) ** dim))))


def jobs(pattern):
    return [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RES, pattern))) if os.path.basename(d) not in EXCLUDE]


def node_class(job):
    names = []
    for fn, col in (("hosts.csv", None), ("nodes.csv", 0)):
        p = os.path.join(RES, job, fn)
        if os.path.exists(p):
            for line in open(p):
                f = line.strip().split(",")
                if f and f[0]:
                    names.append(f[0] if col == 0 else (f[2] if f[1] == "serial" or f[1].isdigit() else f[1]))
    return "cn" if names and all(n.startswith("cn-") for n in names) else "gn"


def summarise(groups, multi, allow_single=False):
    """groups: key -> list of (job, row).  Returns key -> record for published points."""
    out = {}
    for key, jr in groups.items():
        nj = len({j for j, _ in jr})
        if nj < 2 and not allow_single:
            continue
        ok = True
        for _, r in jr:
            it, rp = int(r["iters"]), int(r["repeat"])
            if multi:
                ok &= it == 1000 and rp >= 5
            else:
                want = auto_iters(int(r["dim"]), int(r["n"]))
                ok &= it == want or (it < want and float(r["relres"]) < 1e-12)
        if not ok:
            continue
        t = [ms(r, "time_s") for _, r in jr]; m = quant(t, .5)
        rec = dict(ms_per_iter=m, p25_ms=quant(t, .25), p75_ms=quant(t, .75), values=len(t), jobs=nj,
                   within10=sum(abs(x / m - 1) <= .10 for x in t) / len(t))
        rec["wide"] = rec["within10"] < 2 / 3
        for k, c in (("compute_ms", "t_compute"), ("halo_ms", "t_halo"), ("reduce_ms", "t_reduce")):
            rec[k] = quant([ms(r, c) for _, r in jr], .5)
        r0 = jr[0][1]
        rec.update(nodes=int(r0["nodes"]), n=int(r0["n"]), unknowns=int(r0["N"]), iters=int(r0["iters"]),
                   grid="x".join(r0[a] for a in (("px", "py", "pz") if r0["dim"] == "3" else ("px", "py"))))
        cls = [node_class(j) for j, _ in jr]
        rec["cn_gn"] = f"{cls.count('cn')}/{cls.count('gn')}"
        out[key] = rec
    return out


def write(path, recs, lead):
    cols = list(lead) + [k for k in next(iter(recs)) if k not in lead]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recs)
    print(f"wrote {os.path.relpath(path)}: {len(recs)} points")


# ---- E1 serial baseline
e1 = {}
for j in jobs("e1*"):
    for r in rows(j):
        e1.setdefault((int(r["dim"]), int(r["n"])), []).append(ms(r, "time_s"))
serial = {k: quant(v, .5) for k, v in e1.items() if len(v) >= 2 and (max(v) - min(v)) / (sum(v) / len(v)) <= .10}

# ---- E2 strong scaling: E2a single node (e2s*), E2b 8 x 32 (e2m8*)
recs = []
for exp, pat, multi in (("E2a", "e2s*_r[0-9]*", False), ("E2b", "e2m8*_r[0-9]*", True)):
    g = {}
    for j in jobs(pat):
        for r in rows(j):
            g.setdefault((int(r["dim"]), int(r["n"]), r["decomp"], r["halo"], int(r["ranks"])), []).append((j, r))
    for (dim, n, dc, h, R), rec in sorted(summarise(g, multi).items()):
        base = serial.get((dim, n))
        s = base / rec["ms_per_iter"] if base else None
        recs.append(dict(experiment=exp, dim=dim, decomp=dc, halo=h, ranks=R, serial_ms_per_iter=base,
                         speedup=s, efficiency=s / R if s else None, **rec))
write(os.path.join(HERE, "e2_points.csv"), recs, ("experiment", "dim", "n", "decomp", "halo", "ranks"))

# ---- E8 weak scaling: E8s single node (e8s*), E8m 8 x 32 (e8m*, one job)
recs = []
for exp, pat, multi in (("E8s", "e8s_r[0-9]*", False), ("E8m", "e8m_r[0-9]*", True)):
    g = {}
    for j in jobs(pat):
        for r in rows(j):
            g.setdefault((int(r["dim"]), r["decomp"], r["halo"], r["cg"], int(r["ranks"])), []).append((j, r))
    for (dim, dc, h, cg, R), rec in sorted(summarise(g, multi, allow_single=multi).items()):
        recs.append(dict(experiment=exp, dim=dim, decomp=dc, halo=h, cg=cg, ranks=R, single_job=rec["jobs"] < 2, **rec))
write(os.path.join(HERE, "e8_points.csv"), recs, ("experiment", "dim", "decomp", "halo", "cg", "ranks"))
