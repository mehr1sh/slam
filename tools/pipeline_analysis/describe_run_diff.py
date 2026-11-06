#!/usr/bin/env python3
"""Describe how a re-run differs from a reference run (reporting only; decides nothing).

    python3 tools/pipeline_analysis/describe_run_diff.py REFERENCE_DIR NEW_DIR NAME [NAME ...]

NAME is a file (e.g. pairs.csv) or a directory (e.g. correspondences) relative to both run directories.
Called by run_baseline.sh only after its byte comparison has already found a difference; that comparison
(tolerance 0, every byte) alone decides pass/fail. This script only explains the difference, separating:
  FORMAT     comment lines ('#') added, removed or changed; columns present in only one file; row count
  NUMERICAL  cells of columns present in both files whose values differ: per column the number of differing
             cells, the largest absolute difference and the reference/new values at that cell
  TEXT       differing non-numeric cells, and line differences of non-CSV files
Nothing is ignored: every differing column and every differing comment line is counted. Standard library only.
"""
import csv
import io
import sys
from pathlib import Path


def split(path):
    comments, body = [], []
    for line in Path(path).read_text().splitlines(keepends=True):
        (comments if line.startswith("#") else body).append(line)
    return comments, list(csv.reader(io.StringIO("".join(body))))


def as_float(s):
    try:
        return float(s)
    except ValueError:
        return None


class Acc:
    def __init__(self):
        self.files = self.files_identical = 0
        self.only_ref, self.only_new = [], []
        self.comments_ref = self.comments_new = self.comment_files = 0
        self.cols_only_ref, self.cols_only_new = set(), set()
        self.row_mismatch = []
        self.num = {}   # column -> [count, max_abs, example]
        self.text = {}  # column -> [count, example]
        self.lines = []

    def csv(self, ref, new, label):
        cr, rr = split(ref)
        cn, rn = split(new)
        if cr != cn:
            self.comment_files += 1
            self.comments_ref += len([c for c in cr if c not in cn])
            self.comments_new += len([c for c in cn if c not in cr])
        if not rr or not rn:
            return
        hr, hn = rr[0], rn[0]
        self.cols_only_ref |= set(hr) - set(hn)
        self.cols_only_new |= set(hn) - set(hr)
        if len(rr) != len(rn):
            self.row_mismatch.append(f"{label}: {len(rr) - 1} vs {len(rn) - 1} rows")
        ir, inn = {c: i for i, c in enumerate(hr)}, {c: i for i, c in enumerate(hn)}
        for k, (a, b) in enumerate(zip(rr[1:], rn[1:]), start=1):
            for c in hr:
                if c not in inn or ir[c] >= len(a) or inn[c] >= len(b):
                    continue
                x, y = a[ir[c]], b[inn[c]]
                if x == y:
                    continue
                fx, fy = as_float(x), as_float(y)
                if fx is not None and fy is not None:
                    e = self.num.setdefault(c, [0, -1.0, ""])
                    e[0] += 1
                    if abs(fx - fy) > e[1]:
                        e[1], e[2] = abs(fx - fy), f"{label} row {k}: {x} -> {y}"
                else:
                    e = self.text.setdefault(c, [0, ""])
                    e[0] += 1
                    e[1] = e[1] or f"{label} row {k}: {x!r} -> {y!r}"

    def other(self, ref, new, label):
        a, b = Path(ref).read_text().splitlines(), Path(new).read_text().splitlines()
        n = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        self.lines.append(f"{label}: {n} of {max(len(a), len(b))} lines differ")


def describe(ref_dir, new_dir, name):
    acc, r, n = Acc(), Path(ref_dir) / name, Path(new_dir) / name
    if r.is_dir() or n.is_dir():
        fr = {p.relative_to(r) for p in r.rglob("*") if p.is_file()} if r.is_dir() else set()
        fn = {p.relative_to(n) for p in n.rglob("*") if p.is_file()} if n.is_dir() else set()
        acc.only_ref, acc.only_new = sorted(map(str, fr - fn)), sorted(map(str, fn - fr))
        pairs = [(r / p, n / p, f"{name}/{p}") for p in sorted(fr & fn)]
    elif not r.is_file() or not n.is_file():
        print(f"  {name}: present only in the {'reference' if r.is_file() else 'new run'}")
        return
    else:
        pairs = [(r, n, name)]
    for a, b, label in pairs:
        acc.files += 1
        if a.read_bytes() == b.read_bytes():
            acc.files_identical += 1
        elif a.suffix == ".csv":
            acc.csv(a, b, label)
        else:
            acc.other(a, b, label)
    numerical = bool(acc.num)
    fmt = bool(acc.comment_files or acc.cols_only_ref or acc.cols_only_new or acc.row_mismatch or acc.only_ref or acc.only_new)
    kind = " + ".join(k for k, v in (("FORMAT", fmt), ("NUMERICAL", numerical), ("TEXT", acc.text or acc.lines)) if v) or "none found"
    print(f"  {name}: {acc.files} file(s), {acc.files_identical} byte-identical; differences: {kind}")
    if acc.only_ref or acc.only_new:
        print(f"    FORMAT files only in reference {len(acc.only_ref)} {acc.only_ref[:3]}, only in new run {len(acc.only_new)} {acc.only_new[:3]}")
    if acc.comment_files:
        print(f"    FORMAT comment lines: {acc.comment_files} file(s); {acc.comments_ref} line(s) only in reference, "
              f"{acc.comments_new} only in new run")
    if acc.cols_only_ref or acc.cols_only_new:
        print(f"    FORMAT columns only in reference: {sorted(acc.cols_only_ref) or 'none'}; only in new run: {sorted(acc.cols_only_new) or 'none'}")
    for m in acc.row_mismatch[:5]:
        print(f"    FORMAT row count {m}")
    for c, (k, mx, ex) in sorted(acc.num.items()):
        print(f"    NUMERICAL {c}: {k} cell(s) differ, max |diff| {mx:.3g} ({ex})")
    for c, (k, ex) in sorted(acc.text.items()):
        print(f"    TEXT {c}: {k} cell(s) differ ({ex})")
    for line in acc.lines:
        print(f"    TEXT {line}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    print("  (pass/fail is the byte comparison above, tolerance 0; the following only classifies the difference)")
    for name in sys.argv[3:]:
        describe(sys.argv[1], sys.argv[2], name)
