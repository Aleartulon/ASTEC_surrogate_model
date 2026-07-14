import torch as tc
import numpy as np
import os
import json
import pickle
import h5py
import matplotlib.pyplot as plt
from src.common_functions import load_config
from src.models.AE_NODE.training.data_functions import standard_and_inverse_normalization_field
from src.models.AE_NODE.testing.support_functions import compute_errors, compute_global_errors
from src.dataset_generation.dataset.astec_class import Astec_Dataset
from src.dataset_generation.dataset.support_functions import build_dictionary_of_variables, extract_input_output_bc_variables, extract_time_of_simulation, squeeze_first_dimension
from src.models.AE_NODE.testing.explore_generalization.check import extract_op_from_file

class Baseline_Test:
    """Nearest-neighbour baseline: the 'prediction' for a test trajectory is the ground truth
    of the training trajectory whose operator actions are closest to the test ones.
    No encoder/f/decoder involved."""

    #order must match Model_Test.access_trajectory: scalar, core, vessel, lower plenum, faces
    field_keys = ['dictionary_of_input_variables_1', 'dictionary_of_input_variables_36',
                  'dictionary_of_input_variables_76', 'lower_plenum', 'dictionary_of_input_variables_140']

    def __init__(self, information: dict, config_dataset: dict):

        self.path_to_test_data = information['path_to_test_data']
        self.name_test_file = information['name_test_file']
        self.path_to_model = information['path_to_model']
        self.string_after_saving = information['string_after_saving']

        self.device = tc.device(information['device']) if tc.cuda.is_available() else tc.device("cpu")
        print('Device: ', self.device)
        self.trajectories_to_be_plotted = [str(i) for i in information['trajectories_to_be_plotted']]
        self.baseline_fields_prediction_figures = information.get('baseline_fields_prediction_figures', False)
        self.generate_images_error_per_time_step = information['generate_images_error_per_time_step']
        self.generate_istograms = information["generate_istograms"]
        self.normalize_operator_distance = information.get('normalize_operator_distance', True)

        #build necessary directories to save images
        self.directory_images = self.path_to_model + '/Images/Baseline/'
        self.directory_images_fields_reconstruction_scalar = self.directory_images + 'fields_reconstruction_scalar'
        self.directory_images_fields_reconstruction_2d = self.directory_images + 'fields_reconstruction_2d'
        self.directory_images_fields_reconstruction_faces = self.directory_images + 'fields_reconstruction_faces'
        self.directory_images_errors_fields = self.directory_images + 'errors_reconstruction_fields'
        self.directory_images_global_errors_fields = self.directory_images + 'global_errors_reconstruction_fields'
        for directory in [self.directory_images, self.directory_images_fields_reconstruction_scalar,
                          self.directory_images_fields_reconstruction_2d, self.directory_images_fields_reconstruction_faces,
                          self.directory_images_errors_fields, self.directory_images_global_errors_fields]:
            os.makedirs(directory, exist_ok=True)

        #raw ASTEC simulations and indeces of the candidate trajectories for the nearest-neighbour pool.
        #validation trajectories are included by default: a practitioner doing lookup would use every
        #available simulation, and the model also saw the validation set through model selection
        self.path_to_hdf5 = config_dataset['path_to_hdf5']
        self.include_validation_in_pool = information.get('include_validation_in_pool', True)
        x = list(config_dataset['indeces_training_boundaries'])
        if self.include_validation_in_pool:
            x += list(config_dataset['indeces_validation_boundaries'])
        self.indeces_pool = np.concatenate([np.arange(x[2*i], x[2*i+1]+1, 1) for i in range(int(len(x)/2))])
        self.pool_tag = 'training_validation' if self.include_validation_in_pool else 'training'
        print(f'Nearest-neighbour pool: {self.pool_tag}, {len(self.indeces_pool)} candidate indeces')
        self.minimum_length_acceptable_simulation = config_dataset['minimum_length_acceptable_simulation']
        self.which_normalization = config_dataset['which_normalization']

        #Astec_Dataset gives access to the same construction pipeline used to build the test set
        self.astec_dataset = Astec_Dataset(config_dataset)

        #get normalization, needed to denormalize the test data (always named after the training boundaries)
        indeces_training_boundaries = '_' + '_'.join(str(i) for i in config_dataset['indeces_training_boundaries'])
        with open(f"{self.path_to_test_data}/maxima_or_mean{indeces_training_boundaries}.pkl", 'rb') as file:
            self.maxima_or_mean = pickle.load(file)

        with open(f"{self.path_to_test_data}/minima_or_std{indeces_training_boundaries}.pkl", 'rb') as file:
            self.minima_or_std = pickle.load(file)

        for key in self.maxima_or_mean:
            self.maxima_or_mean[key] = self.maxima_or_mean[key].to(self.device)
            self.minima_or_std[key] = self.minima_or_std[key].to(self.device)

        #get trajectories
        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            self.trajectories = list(f.keys())

        self.training_operator_actions = self.extract_training_operator_actions()
        self.cached_training_trajectory = (None, None) #avoid rebuilding the same neighbour twice in a row
        self.discarded_training_trajectories = set()

    def extract_training_operator_actions(self):
        #cache name carries the pool composition, so switching pool never reuses a stale cache
        cache_path = self.directory_images + f'operator_actions_pool_{self.pool_tag}.pkl'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                dictionary_of_op = pickle.load(f)
            print(f'Loaded operator actions of {len(dictionary_of_op)} pool simulations from {cache_path}')
            return dictionary_of_op

        dictionary_of_op = {}
        for index in self.indeces_pool:
            path_trajectory = f'{self.path_to_hdf5}/{index}.h5'
            if not os.path.exists(path_trajectory):
                print(f'Pool simulation {index} not found in {self.path_to_hdf5}, skipping')
                continue
            dictionary_of_op[str(index)] = extract_op_from_file(path_trajectory)
        print(f'Extracted operator actions of {len(dictionary_of_op)} pool simulations')

        with open(cache_path, 'wb') as f:
            pickle.dump(dictionary_of_op, f)
        return dictionary_of_op

    def rank_training_by_distance(self, operator_actions_test: list):
        indeces = list(self.training_operator_actions.keys())
        operators_training = np.array([self.training_operator_actions[i] for i in indeces])
        operators_test = np.array(operator_actions_test)

        if self.normalize_operator_distance:
            #rescale each operator action to [0,1] over the training pool, so that operators
            #with a large range do not dominate the euclidean distance
            minima = operators_training.min(axis=0)
            ranges = operators_training.max(axis=0) - minima
            ranges[ranges == 0.0] = 1.0
            operators_training = (operators_training - minima) / ranges
            operators_test = (operators_test - minima) / ranges

        distances = np.linalg.norm(operators_training - operators_test, axis=1)
        order = np.argsort(distances)
        return [(indeces[i], distances[i]) for i in order]

    def get_training_trajectory(self, index: str):
        if index in self.discarded_training_trajectories:
            return None
        if self.cached_training_trajectory[0] == index:
            return self.cached_training_trajectory[1]
        fields_and_time = self.build_training_trajectory(index)
        if fields_and_time is None:
            self.discarded_training_trajectories.add(index)
        else:
            self.cached_training_trajectory = (index, fields_and_time)
        return fields_and_time

    def build_training_trajectory(self, index: str):
        #same pipeline used to construct the datasets (see Astec_Dataset.build_dataset), without
        #normalization: the baseline fields are compared against the denormalized test fields
        single_simulation, length_simulation = extract_input_output_bc_variables(self.path_to_hdf5, index, 1)
        if length_simulation < self.minimum_length_acceptable_simulation:
            print(f'Training simulation {index} too short, discarded')
            return None
        single_simulation = self.astec_dataset.make_channels_for_dictionary_per_simulation(single_simulation)
        for i in single_simulation[index]:
            if np.any(single_simulation[index][i] > 1e30):
                print(f'Element too large in {i} of training simulation {index}, discarded')
                return None
        single_simulation = self.astec_dataset.substitute_NaN_with_zeros(single_simulation)
        single_simulation = squeeze_first_dimension(single_simulation)
        reshaped_dict = self.astec_dataset.reshape_dataset(single_simulation[index])
        fields = [reshaped_dict[key].to(dtype=tc.float32, device=self.device).unsqueeze(0) for key in self.field_keys]
        time_training = tc.tensor(extract_time_of_simulation(self.path_to_hdf5, index, 1), dtype=tc.float32, device=self.device)
        if fields[0].size(1) != time_training.size(0):
            raise TypeError(f'Simulation {index}: fields have {fields[0].size(1)} time steps but time grid has {time_training.size(0)}')
        return fields, time_training

    def interpolate_in_time(self, data: tc.tensor, time_source: tc.tensor, time_target: tc.tensor):
        #linear interpolation of data (1, T_source, ...) from its own time grid onto the target one.
        #time grids differ between simulations, so index-wise comparison would misalign physical time
        index_left = tc.searchsorted(time_source, time_target, right=True) - 1
        index_left = index_left.clamp(0, len(time_source) - 2)
        t_left = time_source[index_left]
        t_right = time_source[index_left + 1]
        denominator = t_right - t_left
        weight = tc.where(denominator > 0, (time_target - t_left) / denominator, tc.zeros_like(denominator)).clamp(0.0, 1.0)
        weight = weight.view(1, -1, *([1] * (data.dim() - 2)))
        return data[:, index_left] * (1.0 - weight) + data[:, index_left + 1] * weight

    def access_trajectory(self, trajectory):

        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            dictionary_of_input_variables_1 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_1']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_36 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_36']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_76 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_76']), dtype=tc.float32, device = self.device).unsqueeze(0)
            lower_plenum = tc.tensor(np.array(f[trajectory]['lower_plenum']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_140 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_140']), dtype=tc.float32, device = self.device).unsqueeze(0)
            time = tc.tensor(np.array(f[trajectory]['Time']), dtype=tc.float32, device = self.device)

        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], time

    def test(self):
        matches = {}
        with tc.no_grad():
            print('------------------------- Nearest-neighbour baseline -------------------------')
            for trajectory in self.trajectories:
                fields, time = self.access_trajectory(trajectory)
                #de-normalize to compare in physical units
                fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)

                #find the training trajectory with the closest operator actions
                operator_actions_test = extract_op_from_file(f'{self.path_to_hdf5}/{trajectory}.h5')
                candidates = self.rank_training_by_distance(operator_actions_test)
                fields_and_time = None
                for nearest_index, distance in candidates:
                    fields_and_time = self.get_training_trajectory(nearest_index)
                    if fields_and_time is not None:
                        break
                if fields_and_time is None:
                    raise TypeError('No acceptable training trajectory found')
                baseline_fields, time_training = fields_and_time
                print(f'Test trajectory {trajectory}: nearest training trajectory {nearest_index}, distance {distance:.4f}')

                #trajectories have different time grids and lengths: compare on the overlapping
                #physical time window, interpolating the baseline onto the test time grid
                length_test = fields[0].size(1)
                end_time_test = float(time[-1])
                end_time_training = float(time_training[-1])
                length = int((time <= time_training[-1]).sum())
                print(f'Time window: test ends {end_time_test/3600:.3f} h, training ends {end_time_training/3600:.3f} h -> comparing {length}/{length_test} test time steps')
                fields = [x[:, :length] for x in fields]
                time = time[:length]
                baseline_fields = [self.interpolate_in_time(x, time_training, time) for x in baseline_fields]

                matches[trajectory] = {'nearest_training_trajectory': nearest_index,
                                       'distance': float(distance),
                                       'time_steps_test': int(length_test),
                                       'time_steps_training': int(time_training.size(0)),
                                       'end_time_test_h': end_time_test / 3600.0,
                                       'end_time_training_h': end_time_training / 3600.0,
                                       'compared_time_steps': int(length),
                                       'operator_actions_test': [float(i) for i in operator_actions_test],
                                       'operator_actions_training': [float(i) for i in self.training_operator_actions[nearest_index]]}

                # compute errors
                error_per_trajectory = compute_errors(trajectory, baseline_fields, fields, True)

                #print out errors in files and generate images of errors in time
                self.generate_pictures_errors_field_reconstruction(trajectory, error_per_trajectory, time)

                if self.baseline_fields_prediction_figures and trajectory in self.trajectories_to_be_plotted:
                    self.generate_pictures_fields(trajectory, {trajectory: baseline_fields}, {trajectory: fields}, {trajectory: time})

            with open(self.directory_images + 'baseline_matches.json', 'w') as f:
                json.dump(matches, f, indent=4)

            #compute global errors of the baseline
            compute_global_errors(self.directory_images_errors_fields, self.string_after_saving, self.directory_images_global_errors_fields, generate_istograms = self.generate_istograms, which_prediction = 'Baseline')
            print('-----------------------------------------------------------------------')

    def plot_scalar_values(self, trajectory, Time, baseline_fields, denormalized_fields, shape_index = 0, variable_index = 0, field_name='m_cum_H2', ylabel='m_cum_H2', figsize=(5, 5), fontsize=16):
        plt.figure(figsize=figsize)
        plt.plot(Time[trajectory].cpu()[:]/ 3600.0, baseline_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:],
                label='Nearest-neighbour baseline',linestyle='--',alpha=0.7)
        plt.plot(Time[trajectory].cpu()[:]/ 3600.0, denormalized_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:],
                label='Ground truth', alpha=0.7)
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}', fontsize = fontsize)
        plt.savefig(f'{self.directory_images_fields_reconstruction_scalar}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_core_and_vessel_values(self, trajectory, Time, baseline_fields, denormalized_fields, field_name='state_fuel', shape_index = 0, variable_index=0, time_indices=[0, 100, 1000, 20000], figsize=(20, 8), fontsize=16, faces = False):

        fig, axs = plt.subplots(2, len(time_indices), figsize=figsize)

        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indices):
            all_data.append(baseline_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu())
            all_data.append(denormalized_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu())

        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])

        # Plot with consistent color scale
        for count, i in enumerate(time_indices):
            axs[0, count].imshow(baseline_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu(),
                                vmin=vmin, vmax=vmax, origin='lower')
            im = axs[1, count].imshow(denormalized_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu(),
                                    vmin=vmin, vmax=vmax, origin='lower')

            axs[0, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)
            axs[1, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)

            axs[0, count].axis('off')
            axs[1, count].axis('off')

            if count == 0:
                axs[0, 0].text(-0.05, 0.5, 'Baseline', fontsize=fontsize, fontweight='bold',
                            va='center', ha='right', rotation=90, transform=axs[0, 0].transAxes)
                axs[1, 0].text(-0.05, 0.5, 'Ground truth', fontsize=fontsize, fontweight='bold',
                            va='center', ha='right', rotation=90, transform=axs[1, 0].transAxes)

        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)
        fig.suptitle(f'Trajectory number {trajectory}, {field_name}', fontsize=fontsize)
        if not faces:
            plt.savefig(f'{self.directory_images_fields_reconstruction_2d}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.directory_images_fields_reconstruction_faces}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_pictures_fields(self, trajectory_to_be_plotted:str, baseline_fields_per_trajectory:dict, denormalized_fields_per_trajectory:dict, Time:dict):
        dictionary_of_variables = build_dictionary_of_variables()
        scalar_variables = list(dictionary_of_variables['dictionary_of_input_variables_1'].keys())
        core_variables = list(dictionary_of_variables['dictionary_of_input_variables_36'].keys())
        vessel_variables = list(dictionary_of_variables['dictionary_of_input_variables_76'].keys())
        faces_variables = list(dictionary_of_variables['dictionary_of_input_variables_140'].keys())

        # generate figure of global variables
        for count, name in enumerate(scalar_variables):
            self.plot_scalar_values(trajectory_to_be_plotted, Time, baseline_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 0, variable_index = count, field_name = name, ylabel = name.replace('_', ' '), figsize=(5, 5), fontsize=16)

        time_indeces = [0, int(len(Time[trajectory_to_be_plotted])*0.4), int(len(Time[trajectory_to_be_plotted])*0.8), -2]

        # generate figure of core
        for count, name in enumerate(core_variables):
            self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, baseline_fields_per_trajectory, denormalized_fields_per_trajectory, field_name=name, shape_index = 1, variable_index=count, time_indices=time_indeces, figsize=(10, 8), fontsize=16)

        # generate figure of the vessel
        for count, name in enumerate(vessel_variables):
            self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, baseline_fields_per_trajectory, denormalized_fields_per_trajectory, field_name=name, shape_index = 2, variable_index=count, time_indices=time_indeces, figsize=(10, 8), fontsize=16)

        # generate figure of lower plenum
        for count, name in enumerate(vessel_variables):
            self.plot_scalar_values(trajectory_to_be_plotted, Time, baseline_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 3, variable_index = count, field_name = name[:-7] + '_lower_plenum', ylabel = (name[:-7] + ' lower plenum').replace('_', ' '), figsize=(5, 5), fontsize=16)

        # generate figure of faces
        for count, name in enumerate(faces_variables):
            self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, baseline_fields_per_trajectory, denormalized_fields_per_trajectory, field_name=name, shape_index = 4, variable_index=count, time_indices=time_indeces, figsize=(10, 8), fontsize=16, faces = True)

    def generate_pictures_errors_field_reconstruction(self, trajectory:str, error_per_trajectory:dict, time:tc.tensor):
        dictionary_of_variables = build_dictionary_of_variables()
        scalar_variables = [ key+'_scalar' for key in dictionary_of_variables['dictionary_of_input_variables_1']]
        core_variables = [key+'_core' for key in dictionary_of_variables['dictionary_of_input_variables_36']]
        vessel_variables = [key + '_vessel' for key in dictionary_of_variables['dictionary_of_input_variables_76']]
        lower_plenum_variables = [key[:-7] + '_lower_plenum' for key in dictionary_of_variables['dictionary_of_input_variables_76']]
        faces_variables = [key + '_faces' for key in dictionary_of_variables['dictionary_of_input_variables_140']]
        all_variables = (scalar_variables + core_variables + vessel_variables + lower_plenum_variables + faces_variables)

        dict_of_errors = {}
        list_trajectories = list(error_per_trajectory.keys())

        for metric in error_per_trajectory[list_trajectories[0]]:
            dict_of_errors[metric] = []

        for metric in dict_of_errors:
            if len(metric) > 4 and metric[-4:] == 'step': #WATCH OUT, convention is that the ones in time always end with step
                for arr in error_per_trajectory[trajectory][metric]:
                    for error in arr:
                        for count in range(error.size(-1)):
                            dict_of_errors[metric].append(error[:,count].cpu().numpy())
            else:
                for arr in error_per_trajectory[trajectory][metric]:
                    for error in arr:
                        dict_of_errors[metric]+=tuple(error.cpu().numpy())

        #first deal with global errors per trajectory independent of time-steps
        with open(self.directory_images_errors_fields + f'/{trajectory}_global_errors.txt', 'w') as f:
            head = "Variable name\t"
            for metric in dict_of_errors:
                if not (len(metric) > 4 and metric[-4:] == 'step'):
                    head += metric + '\t'
            head += "\n"
            f.write(head)
            for i in range(len(all_variables)):
                column = str(all_variables[i]) + "\t"
                for metric in dict_of_errors:
                    if not (len(metric) > 4 and metric[-4:] == 'step'):
                        column += str(dict_of_errors[metric][i]) + '\t'
                column += "\n"
                f.write(column)

        if trajectory in self.trajectories_to_be_plotted:
            #now deal with global errors per trajectory per time-steps
            if self.generate_images_error_per_time_step:
                for count, variable_name in enumerate(all_variables):
                    accepted_variables = ['Q_H20_connection_primary_to_vessel_scalar', 'Q_steam_connection_primary_to_vessel_scalar', 'm_H20_connection_primary_to_vessel_scalar',
                                          'Q_H20_connection_vessel_to_primary_scalar','Q_steam_connection_vessel_to_primary_scalar', 'm_H20_connection_vessel_to_primary_scalar']
                    if variable_name in accepted_variables:

                        for metric in dict_of_errors:
                            if len(metric) > 4 and metric[-4:] == 'step':
                                plt.plot(time.cpu().numpy()/3600, dict_of_errors[metric][count])
                                plt.title(variable_name, fontsize = 16)
                                plt.xlabel('Time, h', fontsize = 16)
                                plt.ylabel(metric.replace("_", " "), fontsize = 16)
                                plt.yscale('log')
                                np.save(f'{self.directory_images_errors_fields}/{trajectory}_{variable_name}_{metric}_Time.npy', time.cpu().numpy()/3600)
                                np.save(f'{self.directory_images_errors_fields}/{trajectory}_{variable_name}_{metric}_Y.npy', dict_of_errors[metric][count])
                                plt.close()

def main():
    print(f"PID process: {os.getpid()}")
    config_test = load_config('configs/config_test.yaml')
    config_dataset = load_config('configs/config_dataset.yaml')
    baseline_test = Baseline_Test(config_test, config_dataset)

    print('---------- Start Baseline Testing ----------')
    for key, value in config_test.items():
        print(key, ' : ', value)

    baseline_test.test()

if __name__ == '__main__':
    main()
