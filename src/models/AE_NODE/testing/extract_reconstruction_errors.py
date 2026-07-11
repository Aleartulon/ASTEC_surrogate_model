#!/usr/bin/env python3
"""
Extract per-time-step reconstruction-error fields saved as .npy files into a
single nested dictionary and save it to disk.

File naming convention (fixed):
    {simulation}_{field}_{metric}_per_time_step_{Time|Y}.npy

e.g.
    1000_Q_steam_connection_primary_to_vessel_scalar_RMSE_divided_by_max_per_time_step_Y.npy

For every file:
    - simulation : leading integer (e.g. 1000 .. 1027)
    - field      : physical variable / connection (e.g. Q_steam_connection_primary_to_vessel_scalar)
    - metric     : one of METRICS below
    - suffix     : "Time" (x-axis) or "Y" (metric value)

Resulting structure:
    data[simulation][field][metric] = {"Time": np.ndarray, "Y": np.ndarray}

Only the input PATH is meant to change; the metrics / suffixes / naming are fixed.

Usage:
    python extract_reconstruction_errors.py                  # use built-in default path
    python extract_reconstruction_errors.py --path /some/dir
    python extract_reconstruction_errors.py --path /some/dir --output ~/ASTEC_surrogate_model/errors.pkl
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Fixed assumptions (only the input PATH is expected to change)
# ---------------------------------------------------------------------------
DEFAULT_PATH = (
    "/scratch/ROM_datasets_ale/ASTEC/saved_logs/SBO/AE_NODE/Models/"
    "NO_SMOOTHING_6_gamma_0.99_reset_lr_0.00045/Images/AE_NODE/"
    "errors_reconstruction_fields"
)

DEFAULT_OUTPUT = os.path.expanduser(
    "~/ASTEC_surrogate_model/reconstruction_errors.pkl"
)

# Metrics that may appear in a filename. Order does NOT matter here: matching is
# always done longest-first so that e.g. "RMSE_divided_by_max" wins over "RMSE"
# and "MSE_normalized_by_mean" wins over "MSE".
METRICS = [
    "L2_error_norm",
    "MSE_normalized_by_mean",
    "MSE",
    "RMSE_divided_by_max",
    "RMSE_divided_by_mean",
    "RMSE_divided_by_std",
    "RMSE",
]

SUFFIXES = ("Time", "Y")

# Longest metric first -> unambiguous suffix matching of "{field}_{metric}".
_METRICS_BY_LEN = sorted(METRICS, key=len, reverse=True)

# {sim}_{rest}_per_time_step_{Time|Y}.npy
_FNAME_RE = re.compile(
    r"^(?P<sim>\d+)_(?P<rest>.+)_per_time_step_(?P<suffix>Time|Y)\.npy$"
)


def parse_filename(filename):
    """Return (sim, field, metric, suffix) or None if the name doesn't match."""
    m = _FNAME_RE.match(filename)
    if m is None:
        return None

    sim = m.group("sim")
    rest = m.group("rest")        # "{field}_{metric}"
    suffix = m.group("suffix")

    for metric in _METRICS_BY_LEN:
        tail = "_" + metric
        if rest.endswith(tail):
            field = rest[: -len(tail)]
            if field:                       # there must be an actual field part
                return sim, field, metric, suffix

    return None


def extract(path):
    """Walk `path` and build data[sim][field][metric] = {"Time": ..., "Y": ...}."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Input path does not exist: {path}")

    # data[sim][field][metric] -> {"Time": arr, "Y": arr}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    n_loaded, n_skipped = 0, 0
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".npy"):
            continue
        parsed = parse_filename(filename)
        if parsed is None:
            n_skipped += 1
            print(f"  [skip] could not parse: {filename}")
            continue

        sim, field, metric, suffix = parsed
        arr = np.load(os.path.join(path, filename), allow_pickle=True)
        data[sim][field][metric][suffix] = arr
        n_loaded += 1

    # Convert nested defaultdicts to plain dicts for clean pickling.
    plain = {
        sim: {
            field: {metric: dict(suf) for metric, suf in fields.items()}
            for field, fields in sims.items()
        }
        for sim, sims in data.items()
    }

    print(f"\nLoaded {n_loaded} arrays, skipped {n_skipped} file(s).")
    return plain


def summarize(data):
    sims = sorted(data)
    print(f"Simulations ({len(sims)}): {', '.join(sims)}")
    if not sims:
        return
    s0 = sims[0]
    fields = sorted(data[s0])
    print(f"Fields per simulation ({len(fields)}):")
    for f in fields:
        print(f"    - {f}")
    metrics = sorted(data[s0][fields[0]])
    print(f"Metrics per field ({len(metrics)}): {', '.join(metrics)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-time-step reconstruction-error .npy fields "
        "into a nested dict {sim: {field: {metric: {Time, Y}}}}."
    )
    parser.add_argument(
        "--path", default=DEFAULT_PATH,
        help="Directory containing the .npy files (default: built-in path).",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output pickle file (default: ~/ASTEC_surrogate_model/reconstruction_errors.pkl).",
    )
    args = parser.parse_args()

    path = os.path.expanduser(args.path)
    output = os.path.expanduser(args.output)

    print(f"Reading from: {path}")
    data = extract(path)
    summarize(data)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved nested dictionary to: {output}")


if __name__ == "__main__":
    main()
