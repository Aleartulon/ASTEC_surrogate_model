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

    for which_division in DIVISIONS:
        suffix = f'_maximum_fuel_temperature_RMSE_divided_by_{which_division}_per_time_step.npy'
        error_files = sorted(glob.glob(os.path.join(args.errors_dir, '*' + suffix)))
        if not error_files:
            raise FileNotFoundError(f'No *{suffix} files found in {args.errors_dir}')

        plt.figure(figsize=(10, 5))
        for error_file in error_files:
            trajectory = os.path.basename(error_file)[:-len(suffix)]
            error = np.load(error_file)
            time_hours = np.load(os.path.join(args.errors_dir, f'{trajectory}_maximum_fuel_temperature_Time.npy'))
            plt.plot(time_hours, error, label=f'Trajectory {trajectory}', alpha=0.7)

        plt.title('Maximum fuel temperature', fontsize=args.fontsize)
        plt.xlabel('Time, h', fontsize=args.fontsize)
        plt.ylabel(f'RMSE divided by {which_division}', fontsize=args.fontsize)
        plt.yscale('log')
        if len(error_files) <= 20:
            plt.legend(fontsize=9, ncol=2)
        plt.savefig(os.path.join(args.where_to_save, f'all_trajectories_maximum_fuel_temperature_RMSE_divided_by_{which_division}_per_time_step.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved RMSE divided by {which_division} figure ({len(error_files)} trajectories)')

if __name__ == '__main__':
    main()
