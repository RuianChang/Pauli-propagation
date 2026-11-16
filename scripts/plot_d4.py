#!/usr/bin/env python3
"""Plot D=4 simulation CSVs together with the exact thermodynamic-limit curve.

Example:
    python scripts/plot_d4.py --inputs results/variance_average_*.csv \
      --output figures/d4_reproduced.png

Requires matplotlib. Input CSVs must be generated with INITIAL_OPERATOR_MODE=random_pauli
and contain N_LAYER, L, pT, mean_variance, and Ns_completed columns.
If multiple jobs are supplied for a given system size, their means are pooled
by sample counts. No raw CSVs are bundled with the provided screenshot.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from d4_analytic import variance_density_d4_infinite


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('figures/d4_reproduced.png'))
    args = ap.parse_args()

    groups = defaultdict(list)
    jobs_seen = set()
    for path in args.inputs:
        with path.open(newline='') as inp:
            rows = list(csv.DictReader(inp))
        if not rows:
            continue
        for row in rows:
            if int(row['N_LAYER']) != 4:
                continue
            mode = row.get('initial_operator_mode', '')
            if mode != 'random_pauli':
                raise ValueError(f'{path}: requires random_pauli data, got {mode!r}')
            n, p = int(row['L']), float(row['pT'])
            job_key = (n, p, row.get('job_index', ''), str(path.resolve()))
            if job_key in jobs_seen:
                raise ValueError(f'duplicate row in {path}: N={n}, pT={p}')
            jobs_seen.add(job_key)
            samples = int(row['Ns_completed'])
            if samples < 1:
                raise ValueError(f'{path}: Ns_completed must be positive')
            groups[(n, p)].append((float(row['mean_variance']), samples))
    if not groups:
        raise SystemExit('No D=4 rows found in the given input files.')

    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    for n in sorted({n for n, _ in groups}):
        points = []
        for (size, p), values in groups.items():
            if size != n:
                continue
            total = sum(k for _, k in values)
            mean_v = sum(m * k for m, k in values) / total
            points.append((p, mean_v / (n * 4)))
        points.sort()
        ax.plot([p for p, _ in points], [v for _, v in points], '-o',
                markersize=4.5, linewidth=1.7, label=f'N = {n}')
    ps = [i / 200 for i in range(201)]
    ax.plot(ps, [variance_density_d4_infinite(p) for p in ps],
            'k--', linewidth=2.5, label=r'Analytic ($N\to\infty$)')
    ax.set(xlabel=r'$p_T$', ylabel=r'$V/(ND)$', title='Variance density (D = 4)')
    ax.grid(alpha=0.22)
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
