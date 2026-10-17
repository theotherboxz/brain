#!/usr/bin/env python3
"""
Script 1: Neuron factory (customizable).

Creates plain neurons, each with a 3D nucleus position. That's all.
Jobs like sensor or motor are NOT built in: your project gives them later
(see brain_roles.py).

Output: neurons.csv
Columns: id, x, y, z
(Only if you define custom types with --types, three more columns are added:
 cell_type, sign, role.)

You can customize it three ways (later ones override earlier ones):
  1. Defaults
  2. A JSON config file:   --config my_brain.json
  3. Command-line flags

Examples:
  python factory_brain.py
  python factory_brain.py --n 8000 --shape sphere --size 120
  python factory_brain.py --shape ellipsoid --size 160 80 60
  python factory_brain.py --types "eye=10:+:input:back,mid=60:+:hidden:core,brake=20:-:hidden,leg=10:+:output:front"
  python factory_brain.py --write-config my_brain.json     (saves a template to edit)
  python factory_brain.py --config my_brain.json

Shapes: box, sphere, ellipsoid, cylinder
Type format: name=fraction[:sign[:role[:zone]]]
  sign : + (excitatory) or - (inhibitory)
  role : input (receives nothing), hidden, output (sends nothing)
  zone : any, core, shell, left, right, top, bottom, front, back
"""
import argparse
import csv
import json
import sys

import numpy as np

SHAPES = ("box", "sphere", "ellipsoid", "cylinder")
ZONES = ("any", "core", "shell", "left", "right", "top", "bottom", "front", "back")
ROLES = ("input", "hidden", "output")

DEFAULT_TYPES = [
    {"name": "neuron", "fraction": 1.0, "sign": 1, "role": "hidden", "zone": "any"},
]
DEFAULTS = {
    "n": 5000,
    "shape": "box",
    "size": [100.0, 100.0, 100.0],
    "seed": 42,
    "out": "neurons.csv",
    "types": DEFAULT_TYPES,
}


# ---------- parsing and validation ----------

def parse_sign(v):
    s = str(v).strip().lower()
    if s in ("+", "1", "+1", "exc", "excitatory"):
        return 1
    if s in ("-", "-1", "inh", "inhibitory"):
        return -1
    raise ValueError(f"bad sign {v!r}: use + or -")


def parse_types_string(spec):
    types = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad type {item!r}: use name=fraction[:sign[:role[:zone]]]")
        name, rest = item.split("=", 1)
        parts = rest.split(":")
        types.append({
            "name": name.strip(),
            "fraction": float(parts[0]),
            "sign": parts[1] if len(parts) > 1 and parts[1] else "+",
            "role": parts[2] if len(parts) > 2 and parts[2] else "hidden",
            "zone": parts[3] if len(parts) > 3 and parts[3] else "any",
        })
    return types


def clean_types(types):
    if not types:
        raise ValueError("need at least one cell type")
    out = []
    for t in types:
        name = str(t.get("name", "")).strip()
        if not name or "," in name:
            raise ValueError(f"bad cell type name {name!r}")
        fraction = float(t.get("fraction", 1))
        if fraction <= 0:
            raise ValueError(f"{name}: fraction must be above 0")
        role = str(t.get("role", "hidden")).lower()
        zone = str(t.get("zone", "any")).lower()
        if role not in ROLES:
            raise ValueError(f"{name}: role must be one of {', '.join(ROLES)}")
        if zone not in ZONES:
            raise ValueError(f"{name}: zone must be one of {', '.join(ZONES)}")
        out.append({"name": name, "fraction": fraction,
                    "sign": parse_sign(t.get("sign", 1)), "role": role, "zone": zone})
    return out


def clean_size(size, shape):
    if isinstance(size, (int, float)):
        size = [size]
    size = [float(s) for s in size]
    if len(size) == 1:
        size = size * 3
    if len(size) != 3 or min(size) <= 0:
        raise ValueError("size needs 1 or 3 positive numbers")
    if shape == "sphere":
        size = [size[0]] * 3
    return size


# ---------- geometry ----------
# Points are generated in a normalized space u in [-1, 1]^3, then scaled to the size.
# x = left/right, y = bottom/top, z = back/front.

def in_shape(u, shape):
    if shape == "box":
        return np.ones(len(u), dtype=bool)
    if shape in ("sphere", "ellipsoid"):
        return (u ** 2).sum(axis=1) <= 1.0
    if shape == "cylinder":                       # axis runs along y
        return (u[:, 0] ** 2 + u[:, 2] ** 2) <= 1.0
    raise ValueError(f"unknown shape {shape!r}")


def in_zone(u, zone):
    r = np.sqrt((u ** 2).sum(axis=1))
    if zone == "any":    return np.ones(len(u), dtype=bool)
    if zone == "core":   return r <= 0.5
    if zone == "shell":  return r >= 0.7
    if zone == "left":   return u[:, 0] < -0.3
    if zone == "right":  return u[:, 0] > 0.3
    if zone == "bottom": return u[:, 1] < -0.3
    if zone == "top":    return u[:, 1] > 0.3
    if zone == "back":   return u[:, 2] < -0.3
    if zone == "front":  return u[:, 2] > 0.3
    raise ValueError(f"unknown zone {zone!r}")


def sample_u(shape, zone, k, rng):
    got, have = [], 0
    for _ in range(300):
        cand = rng.uniform(-1, 1, size=(max(2000, k * 4), 3))
        keep = cand[in_shape(cand, shape) & in_zone(cand, zone)]
        got.append(keep)
        have += len(keep)
        if have >= k:
            break
    else:
        raise ValueError(f"zone '{zone}' has no room inside a {shape}")
    return np.concatenate(got)[:k]


def allocate(n, fractions):
    f = np.array(fractions, dtype=float)
    f = f / f.sum()
    raw = f * n
    counts = np.floor(raw).astype(int)
    leftover = n - counts.sum()
    counts[np.argsort(-(raw - counts))[:leftover]] += 1
    return counts


# ---------- the factory ----------

def make_neurons(cfg):
    rng = np.random.default_rng(cfg["seed"])
    dims = np.array(cfg["size"], dtype=float)
    types = cfg["types"]
    counts = allocate(cfg["n"], [t["fraction"] for t in types])

    blocks, labels = [], []
    for t, c in zip(types, counts):
        if c == 0:
            continue
        u = sample_u(cfg["shape"], t["zone"], int(c), rng)
        blocks.append((u + 1) / 2 * dims)
        labels += [t] * int(c)

    pos = np.concatenate(blocks)
    perm = rng.permutation(len(pos))              # shuffle so ids aren't grouped by type
    pos = pos[perm]
    labels = [labels[i] for i in perm]

    rows = []
    for i, (p, t) in enumerate(zip(pos, labels)):
        rows.append({"id": i,
                     "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4),
                     "cell_type": t["name"], "sign": t["sign"], "role": t["role"]})
    return rows, counts


def build_config(args):
    cfg = json.loads(json.dumps(DEFAULTS))        # deep copy
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for key in ("n", "shape", "seed", "out"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if args.size is not None:
        cfg["size"] = args.size
    if args.types is not None:
        cfg["types"] = parse_types_string(args.types)

    cfg["shape"] = str(cfg["shape"]).lower()
    if cfg["shape"] not in SHAPES:
        raise ValueError(f"shape must be one of {', '.join(SHAPES)}")
    if int(cfg["n"]) < 1:
        raise ValueError("n must be at least 1")
    cfg["n"] = int(cfg["n"])
    cfg["size"] = clean_size(cfg["size"], cfg["shape"])
    cfg["types"] = clean_types(cfg["types"])
    return cfg


def main():
    p = argparse.ArgumentParser(description="Generate neurons with 3D positions.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--write-config", metavar="FILE", help="save the settings to a JSON file and exit")
    p.add_argument("--n", type=int, help="number of neurons (default 5000)")
    p.add_argument("--shape", help="box, sphere, ellipsoid, cylinder")
    p.add_argument("--size", type=float, nargs="+", help="one number, or x y z (default 100)")
    p.add_argument("--types", help='e.g. "sensory=10:+:input,inter=70:+:hidden,motor=20:+:output"')
    p.add_argument("--seed", type=int, help="random seed (default 42)")
    p.add_argument("--out", help="output file (default neurons.csv)")
    args = p.parse_args()

    try:
        cfg = build_config(args)
        if args.write_config:
            with open(args.write_config, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"Saved settings to {args.write_config}")
            return
        rows, counts = make_neurons(cfg)
    except (ValueError, OSError, json.JSONDecodeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(2)

    extended = len(cfg["types"]) > 1 or any(
        t["name"] != "neuron" or t["sign"] != 1 or t["role"] != "hidden" for t in cfg["types"])
    fields = ["id", "x", "y", "z"] + (["cell_type", "sign", "role"] if extended else [])
    with open(cfg["out"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    size = " x ".join(f"{s:g}" for s in cfg["size"])
    print(f"Wrote {len(rows)} neurons to {cfg['out']} ({cfg['shape']}, {size})")
    for t, c in zip(cfg["types"], counts if extended else []):
        print(f"  {t['name']}: {int(c)}  sign {'+' if t['sign'] > 0 else '-'}  role {t['role']}  zone {t['zone']}")


if __name__ == "__main__":
    main()
