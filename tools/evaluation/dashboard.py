#!/usr/bin/env python3
"""Local results dashboard: every evaluator run under results/runs/, its metrics, timeline, real correspondences,
RGB-D scene, ICP registration and failure diagnostics. Standard library HTTP server + static frontend.

Usage (repository root; the `results` pixi environment provides numpy / matplotlib / Pillow):
  PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard.py [--port 8765] [--open]
  -> http://127.0.0.1:8765/
  PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard.py --check   # load every run, no server

Data are read from results/runs/ on each request (cached by file modification time): a new stress run or
diagnostic export appears after a page reload, without editing anything. Binds to 127.0.0.1 by default.
"""

import argparse
import json
import mimetypes
import sys
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dashboard_app"))
import data as D  # noqa: E402
import methods as MS  # noqa: E402

STATIC = HERE / "dashboard_app" / "static"


class Handler(BaseHTTPRequestHandler):
    server_version = "slam-dashboard/1"

    def log_message(self, fmt, *args):  # quiet: one line per error only
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, body, ctype="application/json", status=HTTPStatus.OK, cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, base):
        path = path.resolve()
        if base.resolve() not in path.parents or not path.is_file():
            return self._send({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(path.read_bytes(), ctype, cache=base != STATIC)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html", STATIC)
            if u.path.startswith("/static/"):
                return self._file(STATIC / u.path[len("/static/"):], STATIC)
            if u.path.startswith("/output/"):
                return self._file(D.OUTPUT / u.path[len("/output/"):], D.OUTPUT)
            if u.path == "/api/runs":
                return self._send(D.runs_overview())
            if u.path == "/api/run":
                return self._send(D.run_summary(q["id"]))
            if u.path == "/api/timeline":
                return self._send(D.timeline(q["id"]))
            if u.path == "/api/pair":
                return self._send(D.pair(q["id"], int(q["k"])))
            if u.path == "/api/registration":
                return self._send(D.registration(q["id"], int(q["k"])))
            if u.path == "/api/scene":
                extra = tuple(int(x) for x in q.get("extra", "").split(",") if x)
                return self._send(D.scene(q["id"], extra))
            if u.path == "/api/compare":
                return self._send(D.compare(q["a"], q["b"]))
            if u.path == "/api/groups":
                return self._send(D.groups())
            if u.path == "/api/sources":
                return self._send({"sources": MS.discover(), "configurations": MS.configurations(),
                                   "labels": MS.LABELS, "colors": MS.COLORS})
            if u.path == "/api/source_compare":
                return self._send(MS.compare([x for x in q["ids"].split(",") if x]))
            if u.path == "/api/source_scene":
                extra = tuple(int(x) for x in q.get("extra", "").split(",") if x)
                return self._send(MS.source_scene(q["id"], extra))
            if u.path == "/api/corr_index":
                return self._send(MS.corr_index())
            if u.path == "/api/corr":
                return self._send(MS.corr_compare(q["orb"], q["scratch"], int(q["k"])))
            if u.path == "/api/pair_compare":
                return self._send(MS.pair_compare(q["ref"], q["scratch"], int(q["k"])))
            if u.path == "/img/rgb":
                p = D.frame_path(q["id"], int(q["k"]), "rgb")
                return self._send(p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/png", cache=True)
            if u.path == "/img/depth":
                png, _ = D.depth_png(q["id"], int(q["k"]), q.get("mode", "color"))
                return self._send(png, "image/png", cache=True)
            return self._send({"error": f"no route {u.path}"}, status=HTTPStatus.NOT_FOUND)
        except KeyError as e:
            return self._send({"error": f"missing or unknown: {e}"}, status=HTTPStatus.NOT_FOUND)
        except MS.SourceError as e:
            return self._send({"error": str(e)}, status=HTTPStatus.CONFLICT)
        except FileNotFoundError as e:
            return self._send({"error": str(e)}, status=HTTPStatus.NOT_FOUND)
        except Exception as e:  # report, never hide
            traceback.print_exc()
            return self._send({"error": f"{type(e).__name__}: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def check():
    """Load every run through the same functions the server uses; exit 1 on any error."""
    o = D.runs_overview()
    bad = len(o["errors"])
    print(f"{len(o['runs'])} runs, {len(o['incomplete'])} incomplete, {bad} errors")
    for r in o["runs"]:
        m = r["meta"]
        try:
            D.timeline(m["id"])
            src = m["export_source"]
            print(f"  {m['id']:<62} frames {r['metrics']['frames']:>4}  stride {m['stride']}  "
                  f"exports: {'own' if src == m['id'] else ('via ' + src) if src else 'none'}")
            if src:
                k = (D._run_meta(src)["exports"] or [None])[0]
                assert D.pair(m["id"], k)["export"]["available"]
        except Exception as e:
            bad += 1
            print(f"  ERROR {m['id']}: {type(e).__name__}: {e}")
    for i in o["incomplete"]:
        print(f"  incomplete {i['id']}: missing {i['missing']}")
    for e in o["errors"]:
        print(f"  ERROR {e['id']}: {e['error']}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    ap.add_argument("--check", action="store_true", help="load every run and exit (no server)")
    a = ap.parse_args()
    if a.check:
        return check()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"SLAM results dashboard: {url}  (reading {D.OUTPUT}; Ctrl+C to stop)", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
