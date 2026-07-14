"""Plot the maximum fuel temperature errors over time of all test trajectories on the
same figure, one figure per metric (RMSE divided by max, mean and std).

Reads the ``.npy`` files produced by ``Model_Test.maximum_fuel_temperature_errors``
(saved in ``<model>/Images/AE_NODE/maximum_fuel``) and saves the three figures in
``max_fuel_temperature`` next to this script (or in ``--where_to_save``).

Example:
python src/models/AE_NODE/testing/plot_max_fuel_temperature_errors.py \
    --errors_dir .../Images/AE_NODE/maximum_fuel
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIVISIONS = ('max', 'mean', 'std')

def main():
    parser = argparse.ArgumentParser(description='Plot maximum fuel temperature errors over time of all trajectories together')
    parser.add_argument('--errors_dir', type=str, required=True,
                        help='maximum_fuel directory produced by the testing pipeline, e.g. <model>/Images/AE_NODE/maximum_fuel')
    parser.add_argument('--where_to_save', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_fuel_temperature'))
    parser.add_argument('--fontsize', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.where_to_save, exist_ok=True)

    time_suffix = '_maximum_fuel_temperature_Time.npy'
    sims = sorted((os.path.basename(f)[:-len(time_suffix)]
                   for f in glob.glob(os.path.join(args.errors_dir, '*' + time_suffix))),
                  key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    if not sims:
        raise FileNotFoundError(f'No *{time_suffix} files found in {args.errors_dir}')

    #one consistent color per simulation across every figure, as in plot_aggregated_errors
    cmap = plt.get_cmap('turbo')
    colors = {s: cmap(i / max(len(sims) - 1, 1)) for i, s in enumerate(sims)}

    for which_division in DIVISIONS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in sims:
            error = np.load(os.path.join(args.errors_dir, f'{s}_maximum_fuel_temperature_RMSE_divided_by_{which_division}_per_time_step.npy'))
            time_hours = np.load(os.path.join(args.errors_dir, f'{s}{time_suffix}'))
            ax.plot(time_hours, error, lw=0.7, alpha=0.7, color=colors[s], label=s)

        ax.set_xlabel('Time, h', fontsize=args.fontsize)
        ax.set_ylabel(f'RMSE$^i_{{{which_division}}}(t)$', fontsize=args.fontsize)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

        #one shared legend with the simulation numbers, outside the panel, as in plot_aggregated_errors
        handles = [plt.Line2D([0], [0], color=colors[s], lw=2) for s in sims]
        fig.tight_layout(rect=(0, 0, 0.85, 1.0))
        fig.legend(handles, sims, title=r'Simulation $i$', loc='center left',
                   bbox_to_anchor=(0.85, 0.5), ncol=2, frameon=False, borderaxespad=0.0)

        fig.savefig(os.path.join(args.where_to_save, f'all_trajectories_maximum_fuel_temperature_RMSE_divided_by_{which_division}_per_time_step.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved RMSE divided by {which_division} figure ({len(sims)} trajectories)')

if __name__ == '__main__':
    main()