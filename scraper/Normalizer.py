#!/usr/bin/env python
"""
normalize_jdx.py

Reads every .jdx in jdx/, applies absorbance+axis normalization,
and writes out .npz (or .csv) into jdx/normalized/.
"""

import os
import glob
import numpy as np


def parse_xydata_block(lines):
    """
    Given lines of a ##XYDATA=(X++(Y..Y)) block, parse X and multiple Y values,
    and return arrays of (x, y_avg).
    """
    xs, ys = [], []
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0].startswith('##'):
            break
        # First token is X, rest are Y values
        try:
            x_val = float(parts[0])
            y_vals = [float(y) for y in parts[1:]]
        except ValueError:
            continue
        # average Y values
        y_avg = sum(y_vals) / len(y_vals)
        xs.append(x_val)
        ys.append(y_avg)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def normalize_absorbance_jdx(jdx_path, baseline_correct=True):
    """
    Manually parse the JDX file, extract XYDATA, convert to absorbance,
    baseline-correct, and normalize both axes into [0,1].

    Returns (x_norm, A_norm)
    """
    # Read all lines
    with open(jdx_path, 'r') as f:
        lines = f.readlines()

    # Find ##XYDATA block
    data_block = []
    in_block = False
    for line in lines:
        if line.strip().upper().startswith('##XYDATA'):
            in_block = True
            continue
        if in_block:
            if line.strip().startswith('##'):
                # end of XYDATA block
                break
            data_block.append(line)

    # Parse X and Y
    x, y = parse_xydata_block(data_block)
    if x.size == 0 or y.size == 0:
        print(f"[!] Warning: no data in {jdx_path}")
        return np.array([]), np.array([])

    # Convert %T to fraction
    if np.max(y) > 1.1:
        T = y / 100.0
    else:
        T = y
    # To absorbance, avoid log(0)
    A = -np.log10(np.clip(T, 1e-8, None))

    # Baseline-correct (shift min to zero)
    if baseline_correct:
        A = A - np.min(A)

    # Normalize absorbance to [0,1]
    max_A = np.max(A)
    A_norm = A / max_A if max_A > 0 else A.copy()

    # Normalize wavenumber axis to [0,1]
    min_x, max_x = np.min(x), np.max(x)
    x_norm = (x - min_x) / (max_x - min_x) if max_x > min_x else x - min_x

    return x_norm, A_norm


def main():
    # Paths
    JDX_PATH = '../jdx'
    OUT_DIR  = os.path.join(JDX_PATH, 'normalized')
    os.makedirs(OUT_DIR, exist_ok=True)

    pattern = os.path.join(JDX_PATH, '*.jdx')
    for jdx_file in glob.glob(pattern):
        filename = os.path.basename(jdx_file)
        base, _ = os.path.splitext(filename)
        out_file = os.path.join(OUT_DIR, base + '_norm.npz')
        if os.path.exists(out_file):
            continue

        x_norm, A_norm = normalize_absorbance_jdx(jdx_file)
        if x_norm.size == 0:
            continue
        np.savez(out_file, x=x_norm, y=A_norm)
        print(f"Normalized → {out_file}")


if __name__ == '__main__':
    main()