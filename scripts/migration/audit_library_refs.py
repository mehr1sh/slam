#!/usr/bin/env python3
"""Scans the repository for every reference to the vision library being
removed (vision-library migration, docs/migration/).

It reports, per file:
  - #include lines of the library and `using namespace` of its namespace
  - library API symbols/types used (namespace-qualified, or unqualified in
    files that import the namespace), grouped into the audit categories A-L
  - CMake / package-configuration references (category M)
  - text references in documentation, scripts and data notes (category N)

Scope: every file git tracks or would track (tracked + untracked, not ignored).
Generated/ignored files (build/, .pixi/, results/data, ...) are out of scope.

Usage:
  python3 scripts/migration/audit_library_refs.py            # summary to stdout
  python3 scripts/migration/audit_library_refs.py --markdown  # tables for the audit doc
  python3 scripts/migration/audit_library_refs.py --check     # exit 1 if any reference remains
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "open" + "cv"          # spelled indirectly so this scanner does not report itself
NS = "c" + "v"

CATEGORIES = {
    "A image loading": ["imread", "IMREAD_[A-Z_]+", "CV_LOAD_IMAGE_[A-Z_]+"],
    "B image saving": ["imwrite"],
    "C image processing": ["cvtColor", "resize", "GaussianBlur", "countNonZero", "convertTo", "copyTo",
                           "hconcat", "vconcat", "COLOR_[A-Z0-9_]+"],
    "D feature detection": ["FAST", "FeatureDetector", "ORB"],
    "E feature descriptors": ["DescriptorExtractor"],
    "F feature matching": ["DescriptorMatcher", "BFMatcher", "DMatch", "NORM_HAMMING", "KeyPoint"],
    "G PnP": ["solvePnP", "solvePnPRansac", "SOLVEPNP_[A-Z]+"],
    "H RANSAC": ["RANSAC", "FM_RANSAC"],
    "I projection": ["projectPoints", "Rodrigues"],
    "J geometry": ["findFundamentalMat", "findEssentialMat", "findHomography", "recoverPose",
                   "triangulatePoints", "computeCorrespondEpilines", "CV_FM_8POINT", "FM_8POINT"],
    "K matrix operations / core types": ["Mat", "Mat_", "Matx[0-9a-z]*", "Point", "Point2[fdi]", "Point3[fdi]",
                                         "Vec[234][bfdi]", "Scalar", "Size", "Rect", "SVD", "norm", "trace",
                                         "CV_64F", "CV_32F", "CV_8UC[1-4]", "CV_16UC1", "CV_32FC1", "CV_PI",
                                         "RNG", "Ptr", "eigen2cv", "cv2eigen"],
    "L visualization": ["drawMatches", "drawKeypoints", "circle", "line", "putText", "imshow", "waitKey"],
}
SOURCE_EXT = {".cpp", ".hpp", ".h", ".cc", ".cxx"}
INCLUDE_RE = re.compile(r"#\s*include\s*[<\"](" + NAME + r"2?/[^>\"]+)[>\"]")
USING_RE = re.compile(r"using\s+namespace\s+" + NS + r"\s*;")


def candidate_files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    files = []
    for f in out:
        if (not f or f.startswith("data/synthetic_bunny/0000") or f.endswith((".png", ".ply", ".pdf", ".blend"))
                or f == "scripts/migration/audit_library_refs.py"):
            continue
        p = ROOT / f
        if p.is_file():
            files.append(f)
    return files


def scan():
    report = {}
    for f in candidate_files():
        text = (ROOT / f).read_text(errors="replace")
        ext = Path(f).suffix
        entry = dict(includes=[], using_ns=False, symbols=defaultdict(set), config=[], text_refs=[])
        if ext in SOURCE_EXT:
            entry["includes"] = sorted(set(INCLUDE_RE.findall(text)))
            entry["using_ns"] = bool(USING_RE.search(text))
            code = re.sub(r"//[^\n]*", "", text)  # comments are documentation, not API use
            for cat, syms in CATEGORIES.items():
                for s in syms:
                    qualified = re.findall(r"\b" + NS + r"::(" + s + r")\b", code)
                    found = set(qualified)
                    if entry["using_ns"]:
                        found |= set(re.findall(r"(?<![\w.:>])(" + s + r")\b(?=\s*[(<>:{&*,;)\]\s])", code))
                    if s.startswith("CV_"):
                        found |= set(re.findall(r"\b(" + s + r")\b", code))
                    if found:
                        entry["symbols"][cat] |= found
            for i, line in enumerate(text.split("\n"), 1):
                if re.search(NAME, line, re.I) and "//" in line and not INCLUDE_RE.search(line):
                    entry["text_refs"].append((i, line.strip()[:120]))
        elif Path(f).name in ("CMakeLists.txt", "pixi.toml", "pixi.lock") or ext == ".cmake":
            for i, line in enumerate(text.split("\n"), 1):
                if re.search(NAME, line, re.I):
                    entry["config"].append((i, line.strip()[:120]))
        else:
            for i, line in enumerate(text.split("\n"), 1):
                if re.search(NAME + r"|\b" + NS + r"::|\b" + NS + r"2\b|solvePnP|imread|imwrite", line, re.I):
                    entry["text_refs"].append((i, line.strip()[:120]))
        if entry["includes"] or entry["using_ns"] or entry["symbols"] or entry["config"] or entry["text_refs"]:
            report[f] = entry
    return report


def markdown(report):
    lines = ["| File | Includes | API / types used (by category) |", "|---|---|---|"]
    for f, e in sorted(report.items()):
        if not (e["includes"] or e["symbols"]):
            continue
        cats = "<br>".join(f"**{c.split()[0]}** {', '.join(sorted(s))}" for c, s in sorted(e["symbols"].items()))
        inc = ", ".join(i.split("/")[-1] for i in e["includes"]) + (" + `using namespace`" if e["using_ns"] else "")
        lines.append(f"| `{f}` | {inc or '—'} | {cats or '—'} |")
    lines += ["", "### Build / package configuration (M)", "", "| File | Line | Text |", "|---|---|---|"]
    for f, e in sorted(report.items()):
        for i, t in e["config"]:
            lines.append(f"| `{f}` | {i} | `{t}` |")
    lines += ["", "### Documentation, scripts, comments, notes (N)", "", "| File | References |", "|---|---|"]
    for f, e in sorted(report.items()):
        if e["text_refs"]:
            lines.append(f"| `{f}` | {len(e['text_refs'])} (lines {', '.join(str(i) for i, _ in e['text_refs'][:12])}"
                         f"{', …' if len(e['text_refs']) > 12 else ''}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    report = scan()
    if a.markdown:
        print(markdown(report))
        return
    n_src = sum(1 for e in report.values() if e["includes"] or e["symbols"] or e["using_ns"])
    n_cfg = sum(len(e["config"]) for e in report.values())
    n_txt = sum(len(e["text_refs"]) for e in report.values())
    cats = defaultdict(set)
    for f, e in report.items():
        for c in e["symbols"]:
            cats[c].add(f)
    print(f"source files using the library: {n_src}; configuration lines: {n_cfg}; text references: {n_txt}")
    for c in sorted(cats):
        print(f"  {c}: {len(cats[c])} files")
    if a.check:
        sys.exit(1 if report else 0)


if __name__ == "__main__":
    main()
