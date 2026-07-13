#!/usr/bin/env python3
"""
Spaghetti plots of the per-time-step reconstruction-error metrics: every
simulation overlaid on a common absolute-time axis.

No aggregation / no resampling: each simulation is drawn on its own time grid,
so shorter runs simply end earlier. This is the honest view when the runs have
very different durations (here ~6 h to ~40 h) and different time steps.

Input: the nested dict produced by extract_reconstruction_errors.py
    data[sim][field][metric] = {"Time": np.ndarray, "Y": np.ndarray}

Output: one figure per metric (default), each a grid of the 6 fields, with all
simulations overlaid and colored consistently across panels.

Usage:
    python plot_aggregated_errors.py                       # default pickle + ./figures
    python plot_aggregated_errors.py --input /path/errors.pkl --outdir /path/figs
    python plot_aggregated_errors.py --groupby field       # one figure per field instead
    python plot_aggregated_errors.py --logy                # log y-axis
    python plot_aggregated_errors.py --max-points 0        # disable decimation
"""

import argparse
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})  # all text (labels, ticks, legend) at 16

DEFAULT_INPUT = os.path.expanduser(
    "~/ASTEC_surrogate_model/reconstruction_errors.pkl"
)
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures"
)

# Fixed display order (matches the saved naming); anything else is appended.
METRIC_ORDER = [
    "RMSE",
    "RMSE_divided_by_max",
    "RMSE_divided_by_mean",
    "RMSE_divided_by_std",
    "MSE",
    "MSE_normalized_by_mean",
    "L2_error_norm",
]

# Pretty (mathtext) labels for the metrics, used in titles / axis labels.
METRIC_LABELS = {
    "RMSE": "RMSE",
    "RMSE_divided_by_max": "RMSE$_{max}$",
    "RMSE_divided_by_mean": "RMSE$_{mean}$",
    "RMSE_divided_by_std": "RMSE$_{std}$",
    "MSE": "MSE",
    "MSE_normalized_by_mean": "MSE$_{mean}$",
    "L2_error_norm": "$L_2$ error",
}


def metric_label(metric):
    """LaTeX/mathtext label for a metric (falls back to the raw name)."""
    return METRIC_LABELS.get(metric, metric)


def field_label(field):
    """Short display name, e.g. Q_H20_connection_primary_to_vessel_scalar -> 'Q H2O ptv'."""
    var = field.split("_", 1)[0]                       # Q or m
    fluid = "H2O" if "H20" in field else ("steam" if "steam" in field else "")
    if "primary_to_vessel" in field:
        direction = "ptv"
    elif "vessel_to_primary" in field:
        direction = "vtp"
    else:
        direction = ""
    return " ".join(p for p in (var, fluid, direction) if p)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def ordered(items, preferred):
    items = list(items)
    head = [x for x in preferred if x in items]
    tail = sorted(x for x in items if x not in preferred)
    return head + tail


def decimate(t, y, max_points):
    """Stride-downsample a curve for lighter figures (visual only)."""
    if max_points and len(t) > max_points:
        step = int(math.ceil(len(t) / max_points))
        return t[::step], y[::step]
    return t, y


def grid_shape(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def spaghetti(data, groupby, outdir, logy, max_points):
    sims = sorted(data)
    all_fields = ordered({f for s in data.values() for f in s}, [])
    all_metrics = ordered(
        {m for s in data.values() for f in s.values() for m in f}, METRIC_ORDER
    )

    # One consistent color per simulation across every panel/figure.
    cmap = plt.get_cmap("turbo")
    colors = {s: cmap(i / max(len(sims) - 1, 1)) for i, s in enumerate(sims)}

    os.makedirs(outdir, exist_ok=True)

    if groupby == "metric":
        outer, inner, outer_kind = all_metrics, all_fields, "metric"
    else:
        outer, inner, outer_kind = all_fields, all_metrics, "field"

    written = []
    for outer_key in outer:
        nrows, ncols = grid_shape(len(inner))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False
        )
        axes_flat = axes.ravel()

        for ax, inner_key in zip(axes_flat, inner):
            field = inner_key if outer_kind == "metric" else outer_key
            metric = outer_key if outer_kind == "metric" else inner_key

            n_curves = 0
            for s in sims:
                rec = data.get(s, {}).get(field, {}).get(metric)
                if not rec or "Time" not in rec or "Y" not in rec:
                    continue
                t = np.asarray(rec["Time"], float)
                y = np.asarray(rec["Y"], float)
                t, y = decimate(t, y, max_points)
                ax.plot(t, y, lw=0.7, alpha=0.7, color=colors[s], label=s)
                n_curves += 1

            inner_label = (metric_label(inner_key) if outer_kind == "field"
                           else field_label(inner_key))
            ax.set_title(inner_label)
            ax.set_xlabel("Time, h")
            ax.set_ylabel(metric_label(metric))
            if logy:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.margins(x=0)

        # Hide any unused axes in the grid.
        for ax in axes_flat[len(inner):]:
            ax.set_visible(False)

        # One shared legend for all simulations (compact, outside the panels).
        handles = [
            plt.Line2D([0], [0], color=colors[s], lw=2) for s in sims
        ]
        fig.tight_layout(rect=(0, 0, 0.88, 1.0))
        # Anchor the legend flush against the right edge of the panel grid.
        fig.legend(
            handles, sims, title="Simulation", loc="center left",
            bbox_to_anchor=(0.88, 0.5), ncol=2, frameon=False,
            borderaxespad=0.0,
        )

        safe = outer_key.replace("/", "_")
        out = os.path.join(outdir, f"spaghetti_by-{outer_kind}_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")

    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Pickle from extract_reconstruction_errors.py.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help="Directory for the output figures.")
    p.add_argument("--groupby", choices=["metric", "field"], default="metric",
                   help="One figure per metric (fields as panels) or vice-versa.")
    p.add_argument("--linear", action="store_true",
                   help="Use a linear y-axis (default is logarithmic).")
    p.add_argument("--max-points", type=int, default=5000,
                   help="Decimate each curve to ~this many points for plotting "
                        "(0 = no decimation).")
    args = p.parse_args()

    data = load(os.path.expanduser(args.input))
    print(f"Loaded {len(data)} simulations from {args.input}")
    written = spaghetti(
        data, args.groupby, os.path.expanduser(args.outdir),
        logy=not args.linear, max_points=args.max_points,
    )
    print(f"\nDone. {len(written)} figure(s) in {os.path.expanduser(args.outdir)}")


if __name__ == "__main__":
    main()
