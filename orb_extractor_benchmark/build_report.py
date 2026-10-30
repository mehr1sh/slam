#!/usr/bin/env python3
"""Build one offline, browsable HTML index from ORB benchmark result folders.

This reads existing CSV and dashboard files. It never invents match images or metrics.
For pairs without a saved correspondence image, it generates a metrics-only detail
page and labels the visualization as missing.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from urllib.parse import quote

from visualization import safe_name

CSS = r'''
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;color:#192737;background:#f5f7fb;line-height:1.5}
*{box-sizing:border-box} body{margin:0} a{color:inherit;text-decoration:none} a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #427dde;outline-offset:3px}
.hero{background:linear-gradient(115deg,#162a42,#1c4564);color:#fff;padding:30px max(20px,calc((100vw - 1360px)/2));}
.hero .eyebrow{font-size:.78rem;text-transform:uppercase;letter-spacing:.16em;color:#9bddde;font-weight:750}
.hero h1{font-size:clamp(1.65rem,3vw,2.55rem);letter-spacing:-.045em;margin:4px 0 6px}
.hero p{max-width:790px;margin:0;color:#d9e5f0;font-size:.94rem}
.summary{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}.metric{background:#ffffff17;border:1px solid #ffffff26;padding:9px 17px;border-radius:12px;min-width:140px}.metric b{display:block;font-size:1.3rem}.metric small{font-size:.76rem;color:#c8dce9}
.layout{max-width:1400px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:22px;padding:26px 18px 80px}
.sidebar{position:sticky;top:14px;align-self:start;border:1px solid #dde4ed;background:white;border-radius:14px;padding:16px}.sidebar h3{font-size:.75rem;text-transform:uppercase;letter-spacing:.11em;color:#66758a;margin:0 0 12px}
.nav-link{display:flex;justify-content:space-between;gap:7px;padding:9px 10px;border-radius:8px;font-weight:600;font-size:.89rem;color:#435169}.nav-link:hover{background:#edf6fa;color:#076d80}
.nav-link span:last-child{background:#edf1f6;color:#5d7184;padding:1px 7px;border-radius:7px;font-size:.72rem;white-space:nowrap}
.content{min-width:0}.toolbar{background:#fff;border:1px solid #dce3eb;border-radius:13px;padding:12px;margin-bottom:22px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar input{background:#f7f9fc;border:1px solid #d5dfe9;border-radius:9px;min-width:220px;flex:1;padding:11px 13px;font:inherit;font-size:.91rem}.toolbar small{font-size:.78rem;color:#61728a}
.dataset{scroll-margin-top:16px;margin-bottom:45px}.dataset-head{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:15px}.dataset h2{font-size:1.55rem;letter-spacing:-.04em;margin:0}.dataset-head p{color:#66778b;margin:2px 0;font-size:.88rem}
.dataset-stats{display:flex;flex-wrap:wrap;gap:9px}.pill{padding:6px 10px;background:#e7f3f5;border-radius:9px;color:#176e80;font-size:.81rem;font-weight:700}
.seq{margin:10px 0;border:1px solid #dbe3ed;border-radius:13px;background:#fff;overflow:hidden}.seq>summary{cursor:pointer;list-style:none;padding:16px 18px;display:flex;justify-content:space-between;gap:10px;align-items:center}.seq>summary::-webkit-details-marker{display:none}.seq>summary:hover{background:#f8fbfd}.seq-title{font-weight:750;color:#23374b}.seq-sub{font-size:.77rem;color:#6b7c8f}.count{color:#6b7c8f;font-size:.8rem;white-space:nowrap}.seq-content{border-top:1px solid #e1e8ee;padding:17px;background:#fafbfd}
.pair-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,270px),1fr));gap:14px}.pair-card{border:1px solid #e0e6ee;background:white;border-radius:12px;overflow:hidden;display:flex;flex-direction:column;transition:border .15s,box-shadow .15s}.pair-card:hover{border-color:#70aab9;box-shadow:0 7px 22px #182d4214}.pair-card .thumbs{height:135px;display:grid;grid-template-columns:1fr 1fr;background:#edf1f5;overflow:hidden}.pair-card .thumbs img{width:100%;height:100%;object-fit:cover;min-width:0}.pair-card .thumbs.one{grid-template-columns:1fr}.pair-card .no-thumb{height:135px;display:grid;place-content:center;gap:5px;text-align:center;color:#718297;background:repeating-linear-gradient(135deg,#f3f6fa,#f3f6fa 14px,#e9eef4 14px,#e9eef4 28px);font-size:.82rem}
.pair-card .info{padding:12px 13px;min-width:0}.pair-card h4{margin:0;font-size:.97rem;color:#26384b}.pair-card p{font-size:.76rem;margin:4px 0 7px;color:#62748a;overflow-wrap:anywhere}.badges{display:flex;gap:5px;flex-wrap:wrap}.badge{border-radius:5px;font-size:.68rem;font-weight:650;padding:4px 6px;color:#335d71;background:#e8f3f6}.badge.warn{background:#fff3df;color:#a46c22}.small-label{display:block;font-size:.69rem;margin-top:8px;color:#6b7b8d}
.empty{padding:24px;border-radius:12px;background:#fff;color:#627189;font-size:.9rem;border:1px dashed #cbd4df}.note{margin-top:20px;padding:15px 18px;background:#fff;border:1px solid #dce3ec;border-left:4px solid #4f9aa8;border-radius:10px;font-size:.86rem;color:#485a70}.note code{color:#1d556a;background:#f1f5f8;padding:1px 4px;border-radius:3px}
.details{max-width:1300px;margin:24px auto;padding:0 16px 60px}.back{display:inline-block;padding:9px 12px;background:#fff;border:1px solid #dde4ed;border-radius:8px;margin-bottom:15px;color:#23647a;font-weight:650}.compare{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,450px),1fr));gap:16px}.detail-panel{background:#fff;border:1px solid #dfe5ed;padding:18px;border-radius:12px;min-width:0}.detail-panel h2{margin:0 0 10px;font-size:1.1rem}.detail-panel table{width:100%;border-collapse:collapse;font-size:.89rem}.detail-panel td{padding:8px;border-top:1px solid #e3e9ef;overflow-wrap:anywhere}.detail-panel td:first-child{color:#5d6c7e;width:49%}.detail-panel img{width:100%;height:auto}.note.warning{border-left-color:#deaa53;background:#fffaf0}
@media(max-width:850px){.layout{grid-template-columns:1fr}.sidebar{position:static}.sidebar nav{display:flex;flex-wrap:wrap;gap:6px}.nav-link{min-width:110px}.hero{padding:23px 18px}.toolbar input{min-width:100%}}
@media(prefers-reduced-motion:reduce){.pair-card{transition:none}}
'''

DATASET_NAMES = {'oxford':'Oxford Affine','hpatches':'HPatches','tum':'TUM RGB-D',
                 'kitti':'KITTI Odometry','synthetic':'Synthetic transformations','folder':'Custom images'}
ORDER = {'synthetic':0,'oxford':1,'hpatches':2,'tum':3,'kitti':4,'folder':5}


def esc(text):
    return html.escape(str(text), quote=True)


def link_relative(target: Path, origin: Path):
    # Browser-safe relative link, works even for spaces in path names.
    relative = os.path.relpath(target.resolve(), origin.resolve())
    return quote(Path(relative).as_posix(), safe='/.-_')


def floatval(row, field):
    try: return float(row[field])
    except (ValueError, TypeError, KeyError): return None


def fmt_pct(values):
    values = [v for v in values if v is not None]
    return f'{mean(values)*100:.1f}%' if values else 'N/A'


def meanval(rows, field):
    return fmt_pct([floatval(r, field) for r in rows])


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def detail_page(out_file, entries, master, dataset, sequence, pair_name):
    """Fallback for historical CSV pairs whose correspondence images were not saved."""
    cards = []
    for method in ('opencv','scratch'):
        r = entries.get(method)
        if not r: continue
        geom = r.get('geometry','')
        fields = [('Keypoints image 1 / 2',f"{r['keypoints_1']} / {r['keypoints_2']}"),
                  ('Raw mutual Hamming matches',r['raw_matches']),('Filtered matches',r['matches'])]
        if geom == 'homography_ground_truth':
            fields += [('GT correct matches',r.get('correct_matches','')),
                       ('GT match precision',fmt_pct([floatval(r,'match_precision')])),
                       ('GT keypoint repeatability',fmt_pct([floatval(r,'repeatability')]))]
        else:
            fields += [('F-RANSAC estimated inliers',r.get('fundamental_ransac_inliers','')),
                       ('F-RANSAC estimated ratio',fmt_pct([floatval(r,'fundamental_ransac_inlier_ratio')]))]
        fields += [('Extraction time image 1',str(r.get('extract_1_ms',''))+' ms'),
                   ('Extraction time image 2',str(r.get('extract_2_ms',''))+' ms'),
                   ('Matching time',str(r.get('match_ms',''))+' ms')]
        rows = ''.join(f'<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>' for key,value in fields)
        cards.append(f'<section class="detail-panel"><h2>{esc("OpenCV" if method=="opencv" else "Scratch")} ORB</h2>'
                     '<p>No saved correspondence visualization for this evaluated pair.</p>'
                     f'<table>{rows}</table></section>')
    out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(dataset)} · {esc(sequence)} · {esc(pair_name)}</title><style>{CSS}</style></head><body>
<header class="hero"><div class="eyebrow">ORB benchmark · archived results</div><h1>{esc(dataset)} / {esc(sequence)} / {esc(pair_name)}</h1><p>Comparison from saved CSV measurements.</p></header>
<main class="details"><a class="back" href="{link_relative(master,out_file.parent)}">← All datasets</a>
<div class="note warning"><strong>Match images unavailable for this pair.</strong> This older run saved only some visualizations. The numerical comparison below is genuine. To create the feature-line images for every pair, rerun the benchmark with <code>--visualizations -1 --max-pairs 0</code>.</div>
<div class="compare" style="margin-top:18px">{''.join(cards)}</div></main></body></html>''',encoding='utf-8')


def collect(root):
    """Find CSVs in results/<dataset>, results/all/<dataset> etc."""
    groups = OrderedDict()
    for csvpath in sorted(root.rglob('pairs.csv')):
        if root not in csvpath.parents: continue
        rows = read_rows(csvpath)
        if not rows: continue
        dataset = rows[0].get('dataset','unknown') or 'unknown'
        run = csvpath.parent
        run_name = run.relative_to(root).as_posix()
        # Preserve distinct runs; avoid silently mixing different settings.
        key = (dataset, run_name)
        items = OrderedDict()
        for r in rows:
            ident = (r.get('sequence','?'),r.get('pair','?'))
            items.setdefault(ident,{})[r.get('extractor','unknown')] = r
        pairs = []
        for idx,((sequence,pair),methods) in enumerate(items.items()):
            dashboard = run/'dashboards'/f'{idx:03d}_{safe_name(sequence)}_{safe_name(pair)}.html'
            pictures = [run/'visualizations'/f'{idx:03d}_{method}_{safe_name(pair)}.jpg'
                        for method in ('opencv','scratch')]
            pairs.append({'sequence':sequence,'pair':pair,'methods':methods,
                          'dashboard':dashboard,'pictures':[p for p in pictures if p.is_file()],
                          'idx':idx,'run':run})
        groups[key]={'dataset':dataset,'run_name':run_name,'run':run,'pairs':pairs,'csv':csvpath}
    return sorted(groups.values(),key=lambda g:(ORDER.get(g['dataset'],100),g['run_name']))


def assemble(results: Path, out: Path):
    groups = collect(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_sections=[]; nav=[]; n_all=0; n_drawn=0
    for group_index,g in enumerate(groups):
        name=DATASET_NAMES.get(g['dataset'],g['dataset'].title())
        pairs=g['pairs']; n_all+=len(pairs)
        drawn=sum(bool(p['pictures']) for p in pairs); n_drawn+=drawn
        unique_sequences=OrderedDict()
        for p in pairs:
            unique_sequences.setdefault(p['sequence'],[]).append(p)
        anchor=f'dataset-{group_index}'
        nav.append(f'<a class="nav-link" href="#{anchor}"><span>{esc(name)}</span><span>{len(pairs)}</span></a>')
        all_rows=[r for pair in pairs for r in pair['methods'].values()]
        cvs=[r for r in all_rows if r.get('extractor')=='opencv']
        scratch=[r for r in all_rows if r.get('extractor')=='scratch']
        geometry='GT homography precision' if g['dataset'] in ('oxford','hpatches','synthetic') else 'Estimated F-RANSAC inlier ratio'
        key_metric='match_precision' if g['dataset'] in ('oxford','hpatches','synthetic') else 'fundamental_ransac_inlier_ratio'
        seq_sections=[]
        for seq_name, seq_pairs in unique_sequences.items():
            cards=[]
            for p in seq_pairs:
                methods=p['methods']
                dash=p['dashboard']
                if not dash.is_file():
                    dash=p['run']/'report_pairs'/f'{p["idx"]:05d}_{safe_name(p["sequence"])}_{safe_name(p["pair"])}.html'
                    detail_page(dash,methods,out,g['dataset'],p['sequence'],p['pair'])
                href=link_relative(dash,out.parent)
                imgs=p['pictures']
                image_html=(f'<div class="thumbs {"one" if len(imgs)==1 else ""}">'
                            + ''.join(f'<img loading="lazy" decoding="async" src="{link_relative(im,out.parent)}" alt="{esc(im.stem)} corresponding points">' for im in imgs)
                            + '</div>') if imgs else '<div class="no-thumb"><strong>No match preview</strong><span>Open saved metrics</span></div>'
                cv=methods.get('opencv',{}); sc=methods.get('scratch',{})
                def method_badge(r, who):
                    if not r:return ''
                    rate=floatval(r,key_metric)
                    val=f'{rate*100:.1f}%' if rate is not None else 'N/A'
                    return f'<span class="badge">{who}: {val}</span>'
                marks=method_badge(cv,'OpenCV')+method_badge(sc,'Scratch')
                if not imgs: marks+='<span class="badge warn">Images not generated</span>'
                source=(cv or sc)
                label=f'{Path(source.get("image_1","image 1")).name} → {Path(source.get("image_2","image 2")).name}'
                cards.append(f'<a class="pair-card" href="{href}" data-search="{esc((name+" "+g["run_name"]+" "+seq_name+" "+p["pair"]+" "+label).lower())}">'
                             +image_html+'<div class="info">'
                             f'<h4>{esc(p["pair"].replace("_to_"," → "))}</h4><p>{esc(label)}</p><div class="badges">{marks}</div>'
                             f'<span class="small-label">{esc(geometry)} · Click to open comparison ↗</span></div></a>')
            seq_sections.append(f'<details class="seq" {"open" if not seq_sections else ""}><summary><span><span class="seq-title">{esc(seq_name)}</span><br><span class="seq-sub">{len(seq_pairs)} evaluated image pair(s)</span></span><span class="count">Expand / collapse ▾</span></summary><div class="seq-content"><div class="pair-grid">{"".join(cards)}</div></div></details>')
        run_sections.append(f'''<section class="dataset" id="{anchor}"><div class="dataset-head"><div><h2>{esc(name)}</h2><p>Results folder: <code>{esc(g['run_name'])}</code> · {len(unique_sequences)} sequence(s)</p></div><div class="dataset-stats"><span class="pill">{len(pairs)} pairs</span><span class="pill">{drawn}/{len(pairs)} visualized</span><span class="pill">OpenCV {meanval(cvs,key_metric)}</span><span class="pill">Scratch {meanval(scratch,key_metric)}</span></div></div>{''.join(seq_sections)}</section>''')
    if not groups:
        run_sections=['<div class="empty">No benchmark results found. Run <code>python benchmark.py ...</code> or <code>python run_all.py --auto-datasets datasets --output results --max-pairs 0 --visualizations -1</code> first.</div>']
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ORB Benchmark · All datasets</title><style>{CSS}</style></head><body>
<header class="hero"><div class="eyebrow">Feature extraction · Research benchmark</div><h1>ORB Performance Dashboard</h1><p>Explore datasets, sequences and every evaluated image pair. Select a comparison card to inspect OpenCV ORB versus your original C++ Scratch ORB, including match lines and numerical metrics where visualizations were saved.</p><div class="summary"><div class="metric"><b>{len(groups)}</b><small>Dataset runs</small></div><div class="metric"><b>{n_all:,}</b><small>Image-pair comparisons</small></div><div class="metric"><b>{n_drawn:,}/{n_all:,}</b><small>Pairs with saved previews</small></div></div></header>
<div class="layout"><aside class="sidebar"><h3>Datasets</h3><nav>{''.join(nav) or '<span>No results yet</span>'}</nav></aside><main class="content">
<div class="toolbar"><input type="search" id="search" aria-label="Search dataset, sequence or pair" placeholder="Search datasets, sequences, pairs or image names…"><small id="count">{n_all:,} pairs shown</small></div>
{''.join(run_sections)}
<div class="note"><strong>How to interpret:</strong> Oxford, HPatches and synthetic use ground-truth homographies; TUM, KITTI and video use estimated fundamental-matrix RANSAC consistency, not ground truth. Only a spatially separated subset of matches is drawn, while numerical metrics use all accepted matches. <strong>Older results:</strong> if an image is absent, the metric-only page is available; rerun with <code>--visualizations -1</code> to render every pair.</div></main></div>
<script>(function(){{let input=document.getElementById('search'),items=[...document.querySelectorAll('.pair-card')],groups=[...document.querySelectorAll('.seq')],datasets=[...document.querySelectorAll('.dataset')],count=document.getElementById('count');let filter=()=>{{const q=input.value.trim().toLowerCase();let visible=0;for(let card of items){{let show=card.dataset.search.includes(q);card.hidden=!show;if(show)visible++;}}for(let section of groups){{let show=[...section.querySelectorAll('.pair-card')].some(x=>!x.hidden);section.hidden=!show;if(q&&show)section.open=true;}}for(let ds of datasets)ds.hidden=![...ds.querySelectorAll('.seq')].some(x=>!x.hidden);count.textContent=visible+' pairs shown';}};input.addEventListener('input',filter);}})();</script></body></html>'''
    # A hidden pair card needs to stop occupying grid cells.
    page=page.replace('.pair-card:hover{','.pair-card[hidden],.seq[hidden],.dataset[hidden]{display:none!important}.pair-card:hover{')
    out.write_text(page,encoding='utf-8')
    print(f'Combined HTML: {out.resolve()} | runs={len(groups)}, pairs={n_all}, pairs with images={n_drawn}')
    return len(groups), n_all, n_drawn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'),help='Parent results directory to scan recursively')
    p.add_argument('--output',type=Path,default=None,help='Path to generated master HTML (default results/index.html)')
    a=p.parse_args()
    root=a.results.expanduser().resolve()
    if not root.is_dir():p.error(f'Results directory does not exist: {root}')
    assemble(root,(a.output or root/'index.html').expanduser().resolve())

if __name__=='__main__':main()
