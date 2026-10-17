#!/usr/bin/env python3
"""
View your brain in the browser with one command.

    python view_brain.py neurons.csv connections.csv roles.csv

It builds a small self-contained page (brain_view.html) and opens it.
connections.csv and roles.csv (from brain_roles.py) are optional.
File order doesn't matter.

Controls: drag to rotate, scroll or pinch to zoom, tap a neuron to see its links.

Options:
    --max-neurons 20000   show at most this many neurons (random sample)
    --max-edges 8000      draw at most this many connections (random sample)
    --out brain_view.html output page
    --no-open             build the page but don't open the browser
"""
import argparse
import csv
import json
import random
import sys
import webbrowser
from pathlib import Path

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Brain view</title>
<style>
:root{color-scheme:light;--text:#151b29;--muted:#566178;--panel:rgba(255,255,255,.82)}
@media (prefers-color-scheme: dark){:root{color-scheme:dark;--text:#e7ecf6;--muted:#93a0ba;--panel:rgba(14,20,32,.82)}}
html,body{margin:0;height:100%;overflow:hidden;font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}
canvas{position:fixed;inset:0;width:100%;height:100%;touch-action:none;cursor:grab}
.box{position:fixed;background:var(--panel);border-radius:10px;padding:8px 12px;backdrop-filter:blur(6px)}
#legend{top:max(10px,env(safe-area-inset-top));left:10px;display:grid;gap:3px}
#legend div{display:flex;align-items:center;gap:8px}
#legend i{width:10px;height:10px;border-radius:50%;flex:none}
#legend b{font-weight:400;color:var(--muted);margin-left:auto;padding-left:10px}
#info{left:10px;right:10px;bottom:max(10px,env(safe-area-inset-bottom));color:var(--muted)}
</style></head><body>
<canvas id="c"></canvas>
<div class="box" id="legend"></div>
<div class="box" id="info">Drag to rotate. Pinch or scroll to zoom. Tap a neuron.</div>
<script>
(()=>{
const D=__DATA__, P=D.p, T=D.t, E=D.e, n=T.length, m=E.length/3;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
const mq=matchMedia('(prefers-color-scheme: dark)');
const PD=['#4cd0f5','#a89bff','#ff6f8a','#ffd166','#7ee787','#ff9e64','#f78bd8','#9ad1ff'];
const PL=['#0b8fbf','#6a5ae0','#d6334f','#c98a00','#2e9e4f','#d9692a','#c2409a','#3b7dd8'];
let BG,ACC,EXC,INH,PAL,RING;
function theme(){const d=mq.matches;
  BG=d?'#0a0f19':'#eef1f7';ACC=d?'#8fb0ff':'#2b59d9';RING=d?'#ffffff':'#151b29';
  EXC=d?'90,200,230':'20,120,150';INH=d?'255,110,140':'205,50,80';PAL=d?PD:PL;
  document.querySelectorAll('#legend i').forEach((e,k)=>e.style.background=PAL[k%8]);}
const byType=D.types.map(()=>[]);T.forEach((t,i)=>byType[t].push(i));
const lg=document.getElementById('legend');
D.types.forEach(t=>{const r=document.createElement('div');const i=document.createElement('i');
  const s=document.createElement('span');s.textContent=t.name;const b=document.createElement('b');b.textContent=t.count.toLocaleString();
  r.append(i,s,b);lg.append(r);});
theme();mq.addEventListener('change',()=>{theme();go();});

const sx=new Float32Array(n),sy=new Float32Array(n),sz=new Float32Array(n);
let W=0,H=0,dpr=1,yaw=.6,pitch=-.35,zoom=1,sel=-1,raf=0;
function size(){dpr=Math.min(devicePixelRatio||1,2);W=innerWidth;H=innerHeight;cv.width=W*dpr;cv.height=H*dpr;go();}
function go(){if(!raf)raf=requestAnimationFrame(draw);}

function project(){
  const cy=Math.cos(yaw),sy_=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),sc=Math.min(W,H)*.42*zoom;
  for(let i=0;i<n;i++){
    const x=P[i*3],y=P[i*3+1],z=P[i*3+2];
    const x1=x*cy+z*sy_, z1=-x*sy_+z*cy, y1=y*cp-z1*sp, z2=y*sp+z1*cp, p=3.4/(3.4-z2);
    sx[i]=W/2+x1*p*sc; sy[i]=H/2+y1*p*sc; sz[i]=z2;
  }
}
function draw(){
  raf=0;
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.globalAlpha=1;ctx.fillStyle=BG;ctx.fillRect(0,0,W,H);
  project();
  if(m){
    const pe=new Path2D(),pi=new Path2D(),a0=sel>=0?.1:.25;
    for(let k=0;k<m;k++){const a=E[k*3],b=E[k*3+1],p=E[k*3+2]?pe:pi;p.moveTo(sx[a],sy[a]);p.lineTo(sx[b],sy[b]);}
    ctx.lineWidth=1;
    ctx.strokeStyle='rgba('+EXC+','+a0+')';ctx.stroke(pe);
    ctx.strokeStyle='rgba('+INH+','+a0+')';ctx.stroke(pi);
    if(sel>=0){const h=new Path2D();
      for(let k=0;k<m;k++){const a=E[k*3],b=E[k*3+1];if(a===sel||b===sel){h.moveTo(sx[a],sy[a]);h.lineTo(sx[b],sy[b]);}}
      ctx.lineWidth=1.8;ctx.strokeStyle=ACC;ctx.stroke(h);}
  }
  const r=Math.max(1.4,Math.min(4.5,75/Math.sqrt(n)))*Math.min(W,H)/420, round=n<=6000;
  byType.forEach((idx,t)=>{
    for(let pass=0;pass<2;pass++){
      const path=new Path2D();
      for(const i of idx){
        if((sz[i]>=0)!==(pass===1))continue;
        if(round){path.moveTo(sx[i]+r,sy[i]);path.arc(sx[i],sy[i],r,0,6.2832);}
        else path.rect(sx[i]-r,sy[i]-r,2*r,2*r);
      }
      ctx.globalAlpha=pass?1:.45;ctx.fillStyle=PAL[t%8];ctx.fill(path);
    }
  });
  ctx.globalAlpha=1;
  if(sel>=0){ctx.lineWidth=2;ctx.strokeStyle=RING;ctx.beginPath();ctx.arc(sx[sel],sy[sel],r+5,0,6.2832);ctx.stroke();}
}

function pick(px,py){
  let best=-1,bd=1e9;
  for(let i=0;i<n;i++){const d=Math.hypot(sx[i]-px,sy[i]-py);if(d<14&&d-sz[i]*4<bd){bd=d-sz[i]*4;best=i;}}
  const info=document.getElementById('info');
  if(best<0||best===sel){sel=-1;info.textContent='Drag to rotate. Pinch or scroll to zoom. Tap a neuron.';}
  else{sel=best;let a=0,b=0;
    for(let k=0;k<m;k++){if(E[k*3]===sel)b++;else if(E[k*3+1]===sel)a++;}
    info.textContent='Neuron '+D.ids[sel]+', '+D.types[T[sel]].name+'. '+a+' incoming, '+b+' outgoing'+(m?' (of the connections shown).':'.');}
  go();
}

const ptr=new Map();let moved=0,d0=0,z0=1;
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);ptr.set(e.pointerId,{x:e.clientX,y:e.clientY});
  if(ptr.size===1)moved=0;
  if(ptr.size===2){const[a,b]=[...ptr.values()];d0=Math.hypot(a.x-b.x,a.y-b.y);z0=zoom;}});
cv.addEventListener('pointermove',e=>{const p=ptr.get(e.pointerId);if(!p)return;
  const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;
  if(ptr.size===1){yaw+=dx*.006;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.006));moved+=Math.abs(dx)+Math.abs(dy);}
  else if(ptr.size===2){const[a,b]=[...ptr.values()];zoom=Math.max(.4,Math.min(8,z0*Math.hypot(a.x-b.x,a.y-b.y)/d0));moved+=10;}
  go();});
const up=e=>{const one=ptr.size===1;ptr.delete(e.pointerId);if(one&&moved<6&&e.type==='pointerup')pick(e.clientX,e.clientY);};
cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',up);
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(8,zoom*Math.exp(-e.deltaY*.0012)));go();},{passive:false});
addEventListener('resize',size);size();
})();
</script></body></html>
"""


def read_rows(path):
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"Can't find {path}. Run this from the folder that has your CSV files.")
    if not rows:
        sys.exit(f"{path} has no data rows.")
    return rows


def build(neurons, conns, roles, max_neurons, max_edges):
    rng = random.Random(1)
    idx = list(range(len(neurons)))
    if len(idx) > max_neurons:
        idx = sorted(rng.sample(idx, max_neurons))
    rows = [neurons[i] for i in idx]

    ids, id_map = [], {}
    for k, r in enumerate(rows):
        rid = (r.get("id") or str(idx[k])).strip()
        ids.append(rid)
        id_map[rid] = k

    xyz = [(float(r["x"]), float(r["y"]), float(r["z"])) for r in rows]
    n = len(xyz)
    c = [sum(p[a] for p in xyz) / n for a in range(3)]
    rmax = max(max(sum((p[a] - c[a]) ** 2 for a in range(3)) ** 0.5 for p in xyz), 1e-9)
    flat = []
    for p in xyz:
        flat += [round((p[a] - c[a]) / rmax, 3) for a in range(3)]

    role_of = {(r.get("id") or "").strip(): (r.get("role") or "neuron").strip() for r in roles} if roles else {}
    labels = []
    for k, r in enumerate(rows):
        if roles:
            labels.append(role_of.get(ids[k], "neuron"))
        else:
            labels.append((r.get("cell_type") or "neuron").strip())
    names = []
    for name in labels:
        if name not in names:
            names.append(name)
    if "neuron" in names:                       # plain neurons always get the first color
        names.remove("neuron")
        names.insert(0, "neuron")
    t = [names.index(name) for name in labels]
    types = [{"name": nm, "count": t.count(k)} for k, nm in enumerate(names)]

    edges, total = [], 0
    if conns:
        for r in conns:
            a, b = id_map.get(r["pre"].strip()), id_map.get(r["post"].strip())
            if a is None or b is None:
                continue
            w = float(r["weight"]) if r.get("weight") not in (None, "") else 1.0
            edges.append((a, b, 1 if w >= 0 else 0))
        total = len(edges)
        if total > max_edges:
            edges = rng.sample(edges, max_edges)
    e_flat = [v for e in edges for v in e]

    data = {"p": flat, "t": t, "types": types, "e": e_flat, "ids": ids}
    return data, n, len(edges), total


def main():
    ap = argparse.ArgumentParser(description="View neurons.csv and connections.csv in the browser.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="neurons.csv and (optionally) connections.csv")
    ap.add_argument("--max-neurons", type=int, default=20000)
    ap.add_argument("--max-edges", type=int, default=8000)
    ap.add_argument("--out", default="brain_view.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    neurons = conns = roles = None
    for path in args.files:
        rows = read_rows(path)
        cols = rows[0].keys()
        if "pre" in cols and "post" in cols:
            conns = rows
        elif {"x", "y", "z"} <= set(cols):
            neurons = rows
        elif "id" in cols and "role" in cols:
            roles = rows
        else:
            sys.exit(f"{path} is not a neurons file (x, y, z), connections file (pre, post) or roles file (id, role).")
    if neurons is None:
        sys.exit("Give me the neurons file too (the one with x, y, z columns).")

    try:
        data, n, shown, total = build(neurons, conns, roles, args.max_neurons, args.max_edges)
    except (KeyError, ValueError) as err:
        sys.exit(f"Could not read the files: {err}")

    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")

    msg = f"Showing {n:,} neurons"
    if conns is not None:
        msg += f" and {shown:,} connections" + (f" (a sample of {total:,})" if shown < total else "")
    print(msg)
    print(f"Page saved: {out.resolve()}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
