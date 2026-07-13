"""Compare the aggregated error metrics of two (or more) testing runs, e.g. the
nearest-neighbour baseline vs AE-NODE, or two different models.

Thin wrapper around ``combine_models_in_one_plot`` in ``more_plots.py``: every run is
identified by its ``global_errors_reconstruction_fields`` directory (produced by
compute_global_errors, which stores ``combined_plots/plotting_data.pkl`` inside it), and the
figures follow the same structure as the ones already saved there. On top of the figures a
summary table with per-group medians, ratios and win counts is written.

Note: ``plotting_data.pkl`` contains the three relative RMSE metrics; those are also the only
metrics guaranteed to be comparable between runs produced by different code versions.

Example:
python src/models/AE_NODE/testing/compare_error_metrics.py \
    --dirs .../Images/Baseline/global_errors_reconstruction_fields \
           .../Images/AE_NODE/global_errors_reconstruction_fields \
    --labels Baseline AE-NODE \
    --where_to_save .../Images/comparison_Baseline_AE_NODE
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

try:
    from src.models.AE_NODE.testing.more_plots import combine_models_in_one_plot, _load_models
except ModuleNotFoundError: #also allow running the file directly from any working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from more_plots import combine_models_in_one_plot, _load_models

GROUPS = ['g', 'v', 'p', 'cr_f']

def summary_table(models_data: dict, metrics: list, where_to_save: str, string_after_saving: str):
    """Per metric and variable group: median error of each run, ratio and win count of every
    run against the first one (the reference), computed on the variables common to all runs."""
    labels = list(models_data.keys())
    reference = labels[0]
    with open(f'{where_to_save}/summary_comparison_{string_after_saving}.txt', 'w') as f:
        header = f"{'Metric':<25} {'Group':<6}"
        for label in labels:
            header += f" {'Median ' + label:<22}"
        for label in labels[1:]:
            header += f" {'Ratio ' + label + '/' + reference:<25} {label + ' better':<15}"
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')

        for metric in metrics:
            per_group_values = {label: [] for label in labels}
            for group in GROUPS:
                #only variables present in every run
                common = set(models_data[reference][group][metric]['labels'])
                for label in labels[1:]:
                    common &= set(models_data[label][group][metric]['labels'])
                if not common:
                    continue
                values = {}
                for label in labels:
                    d = models_data[label][group][metric]
                    lut = dict(zip(d['labels'], d['values']))
                    values[label] = np.array([lut[name] for name in sorted(common)])
                    per_group_values[label].append(values[label])

                row = f"{metric:<25} {group:<6}"
                for label in labels:
                    row += f" {np.median(values[label]):<22.6e}"
                for label in labels[1:]:
                    ratio = np.median(values[label]) / np.median(values[reference])
                    wins = int(np.sum(values[label] < values[reference]))
                    row += f" {ratio:<25.4f} {f'{wins}/{len(common)}':<15}"
                f.write(row + '\n')

            #all groups together
            all_values = {label: np.concatenate(per_group_values[label]) for label in labels}
            row = f"{metric:<25} {'all':<6}"
            for label in labels:
                row += f" {np.median(all_values[label]):<22.6e}"
            for label in labels[1:]:
                ratio = np.median(all_values[label]) / np.median(all_values[reference])
                wins = int(np.sum(all_values[label] < all_values[reference]))
                row += f" {ratio:<25.4f} {f'{wins}/{len(all_values[label])}':<15}"
            f.write(row + '\n')
            f.write('-' * len(header) + '\n')
    print(f'Summary written to {where_to_save}/summary_comparison_{string_after_saving}.txt')

def main():
    parser = argparse.ArgumentParser(description='Compare aggregated error metrics of two or more testing runs')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='global_errors_reconstruction_fields directories (or plotting_data.pkl paths), one per run')
    parser.add_argument('--labels', nargs='+', required=True, help='one legend label per run, e.g. Baseline AE-NODE')
    parser.add_argument('--where_to_save', type=str, required=True)
    parser.add_argument('--string_after_saving', type=str, default='SBO')
    parser.add_argument('--metrics', nargs='+', default=['RMSE_divided_by_mean', 'RMSE_divided_by_max', 'RMSE_divided_by_std'])
    parser.add_argument('--no_suptitle', action='store_true', help='do not draw the title on top of the figures')
    args = parser.parse_args()

    if len(args.dirs) != len(args.labels):
        raise TypeError('Number of labels must match the number of directories')

    model_paths = dict(zip(args.labels, args.dirs))
    os.makedirs(args.where_to_save, exist_ok=True)

    for metric in args.metrics:
        suptitle = None if args.no_suptitle else f"{metric.replace('_', ' ')}, {' vs '.join(args.labels)}"
        combine_models_in_one_plot(model_paths, args.where_to_save, args.string_after_saving,
                                   metric=metric, label_prefix='', suptitle=suptitle)

    models_data = _load_models(model_paths)
    summary_table(models_data, args.metrics, args.where_to_save, args.string_after_saving)

if __name__ == '__main__':
    main()
