#!/usr/bin/env python3
"""Draw direct PNG files; this command never loads a browser."""
import argparse
from render_outputs import frontend, manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--i', type=int, default=15)
    p.add_argument('--j', type=int, default=16)
    a = p.parse_args()
    frontend(a.i, a.j)
    manifest()
