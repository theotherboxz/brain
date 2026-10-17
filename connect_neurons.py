#!/usr/bin/env python3
"""
Script 2: Connector.

Reads neurons.csv and wires neurons together by distance:
    P(connect i -> j) = p_max * exp(-distance / lam)

Closer neurons are more likely to connect. Each connection gets a
weight (random magnitude) signed by the presynaptic neuron's sign.

With plain neurons (id, x, y, z) every neuron can send and receive.
If neurons.csv has a "role" column (custom types), then:
    input  neurons receive no connections
    output neurons send no connections
If the file has no role column, "sensory" is treated as input and
"motor" as output.

Output: connections.csv
Columns: pre, post, weight, distance

Usage:
    python connect_neurons.py --lam 8 --pmax 0.3
"""
import argparse
import csv

import numpy as np

OLD_NAME_ROLES = {"sensory": "input", "motor": "output"}


def load_neurons(path):
    ids, pos, signs, roles = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["id"]))
            pos.append((float(row["x"]), float(row["y"]), float(row["z"])))
            signs.append(int(row.get("sign") or 1))
            role = (row.get("role") or "").strip().lower()
            if not role:
                role = OLD_NAME_ROLES.get((row.get("cell_type") or "").strip().lower(), "hidden")
            roles.append(role)
    return np.array(ids), np.array(pos), np.array(signs), np.array(roles)


def connect(pos, roles, signs, lam, pmax, rng, chunk=500):
    n = len(pos)
    can_send = roles != "output"
    can_receive = roles != "input"
    edges = []

    # Process in chunks so memory stays small for large N
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = pos[start:end, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))          # (chunk, n)

        prob = pmax * np.exp(-dist / lam)
        prob[:, ~can_receive] = 0.0                       # input neurons get no input
        prob[~can_send[start:end], :] = 0.0               # output neurons send nothing
        for k in range(end - start):                      # no self-connections
            prob[k, start + k] = 0.0

        hit = rng.random(prob.shape) < prob
        pre_local, post = np.nonzero(hit)
        for a, b in zip(pre_local, post):
            pre = start + a
            weight = signs[pre] * rng.uniform(0.2, 1.0)
            edges.append((pre, int(b), round(float(weight), 4),
                          round(float(dist[a, b]), 4)))
    return edges


def main():
    p = argparse.ArgumentParser(description="Connect neurons by distance.")
    p.add_argument("--neurons", default="neurons.csv")
    p.add_argument("--out", default="connections.csv")
    p.add_argument("--lam", type=float, default=8.0, help="distance decay length")
    p.add_argument("--pmax", type=float, default=0.3, help="max connect probability")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    ids, pos, signs, roles = load_neurons(args.neurons)
    edges = connect(pos, roles, signs, args.lam, args.pmax, rng)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pre", "post", "weight", "distance"])
        w.writerows(edges)

    n = len(ids)
    print(f"{n} neurons -> {len(edges)} connections "
          f"(avg {len(edges)/n:.1f} per neuron) saved to {args.out}")


if __name__ == "__main__":
    main()
