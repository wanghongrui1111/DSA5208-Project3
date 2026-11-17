#!/usr/bin/env python3
"""Checks the CSVs written by validate.sh and writes validation_summary.md.

Standard library only, so it also runs on cluster login nodes.
  python3 scripts/check_validation.py results/<host>/validation
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# A wrong halo or decomposition changes the error against the exact solution
# by O(1). Classic and Chronopoulos-Gear CG reproduce the serial run to round-
# off (same iteration count, error equal to ~1e-9 relative).
ERR_RTOL = 1e-6
ITER_ABS = 2
# Pipelined CG is sensitive to round-off: the iteration count varies with the
# reduction order and its recurrence residual drifts from the true residual
# (the "residual gap"), so it gets looser bounds.
PIPE_ERR_RTOL = 1e-4
PIPE_ITER_REL = 0.05
PIPE_TRUE_RES = 1e-6


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_convergence(rows, out):
    rows = sorted(rows, key=lambda r: int(r["n"]))
    out.append("## Grid refinement (serial, tol 1e-12)\n")
    out.append("| n | h | iterations | max error | L2 error | observed order |")
    out.append("|---|---|---|---|---|---|")
    ok = True
    prev = None
    for r in rows:
        n, e = int(r["n"]), float(r["err_inf"])
        h = 1.0 / (n + 1)
        order = ""
        if prev:
            p = math.log(prev[1] / e) / math.log(prev[0] / h)
            order = f"{p:.3f}"
            ok &= 1.9 < p < 2.1
        out.append(f"| {n} | {h:.3g} | {r['iters']} | {e:.4e} | {float(r['err_l2']):.4e} | {order} |")
        prev = (h, e)
    out.append("")
    return ok


def check_consistency(rows, out):
    refs = {r["cg"]: r for r in rows if r["tag"] == "reference"}
    groups = defaultdict(list)
    failures = []
    for r in rows:
        if r["tag"] == "reference":
            continue
        ref = refs[r["cg"]]
        derr = abs(float(r["err_inf"]) - float(ref["err_inf"])) / float(ref["err_inf"])
        dit = int(r["iters"]) - int(ref["iters"])
        tres = float(r["true_relres"])
        tol = float(r["tol"])
        if r["cg"] == "pipe":
            ok = (derr < PIPE_ERR_RTOL and abs(dit) <= PIPE_ITER_REL * int(ref["iters"])
                  and tres < PIPE_TRUE_RES)
        else:
            ok = derr < ERR_RTOL and abs(dit) <= ITER_ABS and tres < 10 * tol
        groups[(r["impl"], r["comm"], r["cg"])].append((derr, dit, tres, ok))
        if not ok:
            failures.append(r)
    out.append("## Parallel runs against the serial reference (n = %s)\n" % rows[0]["n"])
    out.append("| implementation | halo | CG variant | runs | serial iterations | iteration range | "
               "max rel. error difference | max true residual | result |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (impl, comm, cg), g in sorted(groups.items()):
        ref = refs[cg]
        its = [int(ref["iters"]) + d for _, d, _, _ in g]
        out.append(f"| {impl} | {comm} | {cg} | {len(g)} | {ref['iters']} | {min(its)}-{max(its)} | "
                   f"{max(x[0] for x in g):.2e} | {max(x[2] for x in g):.2e} | "
                   f"{'PASS' if all(x[3] for x in g) else 'FAIL'} |")
    out.append("")
    for r in failures:
        out.append(f"FAILED: {r['impl']} ranks={r['ranks']} threads={r['threads']} decomp={r['decomp']} "
                   f"comm={r['comm']} gpu_mpi={r['gpu_mpi']} cg={r['cg']} iters={r['iters']} "
                   f"err_inf={r['err_inf']} true_relres={r['true_relres']}")
    return not failures


def check_repro(rows, out):
    """repro must give one bit pattern per configuration, whatever the thread count or run."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["impl"], r["comm"], r["reduce"])].append(r)
    out.append("## Reproducibility (n = %s, same run with 1/2/4 threads, repeated)\n" % rows[0]["n"])
    out.append("| implementation | halo | reduction | runs | distinct final residuals (bits) | result |")
    out.append("|---|---|---|---|---|---|")
    ok = True
    for (impl, comm, reduce), g in sorted(groups.items()):
        distinct = len({r["relres_bits"] for r in g})
        good = distinct == 1 or reduce == "fast"
        ok &= good
        verdict = ("PASS" if distinct == 1 else "FAIL") if reduce == "repro" else "(not guaranteed)"
        out.append(f"| {impl} | {comm} | {reduce} | {len(g)} | {distinct} | {verdict} |")
    out.append("")
    return ok


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = ["# Validation summary\n"]
    ok = check_convergence(read(d / "convergence.csv"), out)
    ok &= check_consistency(read(d / "consistency.csv"), out)
    if (d / "reproducibility.csv").exists():
        ok &= check_repro(read(d / "reproducibility.csv"), out)
    failures = (d / "failures.txt").read_text().splitlines() if (d / "failures.txt").exists() else []
    if failures:
        ok = False
        out.append("## Runs that failed\n")
        out += [f"- `{f}`" for f in failures] + [""]
    out.append(f"**Overall: {'PASS' if ok else 'FAIL'}**\n")
    text = "\n".join(out)
    (d / "validation_summary.md").write_text(text)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
