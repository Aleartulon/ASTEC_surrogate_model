from src.dataset_generation.dataset.support_functions import *
import h5py
import numpy as np
import torch as tc
import pickle
import time
import numpy as np
import random
import gc

class Astec_Dataset():
    def __init__(self , config_dataset: dict):
        
        self.path_to_hdf5 = config_dataset['path_to_hdf5']
        self.where_to_save_data = config_dataset['where_to_save_data']
        self.save_dictionary_per_time_lengths = config_dataset['save_dictionary_per_time_lengths']
        self.which_normalization = config_dataset['which_normalization']
        self.percentile_high_norm_min_max = config_dataset['percentile_high_norm_min_max']
        self.percentile_low_norm_min_max = config_dataset['percentile_low_norm_min_max']
        self.device = tc.device(config_dataset['device'] if tc.cuda.is_available() else 'cpu')
        self.polyorder_smoothing = config_dataset['polyorder_smoothing']
        self.window_length_smoothing = config_dataset['window_length_smoothing']
        self.minimum_length_acceptable_simulation = config_dataset['minimum_length_acceptable_simulation']
        self.smoothing = config_dataset['smoothing']
        self.smoothing_method = config_dataset['smoothing_method'] # default keeps backward compat
        self.weight_denoise_tv_chambolle = config_dataset['weight_denoise_tv_chambolle']
        print('Device: ', self.device)
        self.testing = config_dataset['testing']
        self.indeces_training_boundaries = '_'
        self.indeces_validation_boundaries = '_'
        self.indeces_testing_boundaries = '_'
        
        # construct indeces of trainin validation and test 
        x = config_dataset['indeces_training_boundaries']
        if len(x) > 1:
            self.indeces_training = [np.arange(x[2*i],x[2*i+1]+1,1) for i in range(int(len(x)/2))]
            self.indeces_training = np.concatenate(self.indeces_training)
        else:
            self.indeces_training = config_dataset['indeces_training_boundaries']
        x = config_dataset['indeces_validation_boundaries']
        if len(x) > 1:
            self.indeces_validation = [np.arange(x[2*i],x[2*i+1]+1,1) for i in range(int(len(x)/2))]
            self.indeces_validation = np.concatenate(self.indeces_validation)
        else:
            self.indeces_validation = config_dataset['indeces_validation_boundaries']
        
        x = config_dataset['indeces_testing_boundaries']
        if len(x) > 1:
            self.indeces_testing = [np.arange(x[2*i],x[2*i+1]+1,1) for i in range(int(len(x)/2))]
            self.indeces_testing = np.concatenate(self.indeces_testing)
        else:
            self.indeces_testing = config_dataset['indeces_testing_boundaries']
        
        for i in config_dataset['indeces_training_boundaries']:
            self.indeces_training_boundaries += str(i) + '_'
        self.indeces_training_boundaries = self.indeces_training_boundaries[:-1]
        
        for i in config_dataset['indeces_validation_boundaries']:
            self.indeces_validation_boundaries += str(i) + '_'
        self.indeces_validation_boundaries = self.indeces_validation_boundaries[:-1]
        
        for i in config_dataset['indeces_testing_boundaries']:
            self.indeces_testing_boundaries += str(i) + '_'
        self.indeces_testing_boundaries = self.indeces_testing_boundaries[:-1]
        
        #get number of trajectories per dataset
        self.len_training = len(self.indeces_training)
        self.len_validation = len(self.indeces_validation)
        self.len_testing = len(self.indeces_testing)
        
        self.percentages_sampling_training = config_dataset['percentages_sampling_training']
        self.percentages_sampling_validation = config_dataset['percentages_sampling_validation']
        self.percentages_sampling_testing = config_dataset['percentages_sampling_testing']
        self.subsampling_training = config_dataset['subsampling_training']
        self.subsampling_validation = config_dataset['subsampling_validation']
        self.subsampling_testing = config_dataset['subsampling_testing']
        
        if len(self.percentages_sampling_training) != len(self.subsampling_training) or len(self.percentages_sampling_validation) != len(self.subsampling_validation) or len(self.percentages_sampling_testing) != len(self.subsampling_testing):
            raise TypeError(f"Mismatch in sizes of percentages and subsampling arrays")
        if np.sum(self.percentages_sampling_training) != 1.0 or np.sum(self.percentages_sampling_validation) != 1.0 or np.sum(self.percentages_sampling_testing) != 1.0:
            raise TypeError(f"Sum of percentages is not 1.0")
        
        self.subsampling_indeces_training = build_total_sampling_percentages(self.percentages_sampling_training, self.subsampling_training, self.len_training)
        self.subsampling_indeces_validation = build_total_sampling_percentages(self.percentages_sampling_validation, self.subsampling_validation, self.len_validation)
        self.subsampling_indeces_testing = build_total_sampling_percentages(self.percentages_sampling_testing, self.subsampling_testing, self.len_testing)
        
        diff_length_training = self.len_training - len(self.subsampling_indeces_training)
        diff_length_validation = self.len_validation - len(self.subsampling_indeces_validation) 
        diff_length_testing = self.len_testing - len(self.subsampling_indeces_testing) 
        
        self.subsampling_indeces_training = fix_total_percentages(self.subsampling_indeces_training, diff_length_training)
        self.subsampling_indeces_validation = fix_total_percentages(self.subsampling_indeces_validation, diff_length_validation)
        self.subsampling_indeces_testing = fix_total_percentages(self.subsampling_indeces_testing, diff_length_testing)
        
        diff_length_training = self.len_training - len(self.subsampling_indeces_training) 
        diff_length_validation = self.len_validation - len(self.subsampling_indeces_validation) 
        diff_length_testing = self.len_testing - len(self.subsampling_indeces_testing) 
        
        self.subsampling_indeces_training = random.sample(self.subsampling_indeces_training, len(self.subsampling_indeces_training))
        self.subsampling_indeces_validation = random.sample(self.subsampling_indeces_validation, len(self.subsampling_indeces_validation))
        self.subsampling_indeces_testing = random.sample(self.subsampling_indeces_testing, len(self.subsampling_indeces_testing))
        
        if diff_length_training != 0.0 or diff_length_validation != 0.0 or diff_length_testing != 0.0:
            raise TypeError(f"Mismatch in size between total number of simulations and number of indeces of percentages")
        print('subsampling_indeces_training',  self.subsampling_indeces_training)
        print('subsampling_indeces_validation',  self.subsampling_indeces_validation)
        print('subsampling_indeces_testing',  self.subsampling_indeces_testing)
        
    def build_dataset(self, indeces:list, purpose_of_data:str, subsampling_indeces:list):
        
        if purpose_of_data == 'training':
            self.path_to_constructed_data = f"{self.where_to_save_data}/data_training{self.indeces_training_boundaries}.h5"
        elif purpose_of_data == 'validation':
            self.path_to_constructed_data = f"{self.where_to_save_data}/data_validation{self.indeces_validation_boundaries}.h5"
        elif purpose_of_data == 'testing':
            self.path_to_constructed_data = f"{self.where_to_save_data}/data_testing{self.indeces_testing_boundaries}.h5"
            
        # create dictionary and hdf5 file
        self.dictionary_per_simulation = {}
        skipped_simulations = []
        with h5py.File(self.path_to_constructed_data, 'w') as f:
            dict_to_hdf5(self.dictionary_per_simulation, f)
        for count, index_simulation in enumerate(indeces):
            index_simulation = str(index_simulation)
            t1 = time.time()
            single_simulation, length_simulation = extract_input_output_bc_variables(self.path_to_hdf5, index_simulation, subsampling_indeces[count]) #build dictionary of data divided by number of simulation
            if length_simulation < self.minimum_length_acceptable_simulation:
                print("Something wrong with this simulation, too short, skipping")
                skipped_simulations.append(index_simulation)
                continue
            t2 = time.time()
            print(f'Simulation: {index_simulation}. Build dictionary of data divided by number of simulation: {t2-t1} seconds')
            single_simulation = self.make_channels_for_dictionary_per_simulation(single_simulation) #build dictionary of data divided by simulations and make channels per spatial domain
            t3 = time.time()
            skip_simulation = False
            for i in single_simulation[index_simulation]:
                if np.any(single_simulation[index_simulation][i] > 1e30):
                    print(f'Element too large in {i} of simulation {index_simulation}, skipping this simulation')
                    skip_simulation = True
            if skip_simulation:
                skipped_simulations.append(index_simulation)
                continue
            print(f'Simulation: {index_simulation}. Build dictionary of data divided by simulations and make channels per spatial domain: {t3-t2} seconds')
            single_simulation = self.substitute_NaN_with_zeros(single_simulation) #substitute with zeros the NaN values
            t4 = time.time()
            print(f'Simulation: {index_simulation}. Substitute with zeros the NaN values: {t4-t3} seconds')
            #apply smoothing
            if self.smoothing:
                t5 = time.time()
                single_simulation = self.use_smoothing(single_simulation, index_simulation)
                t6 = time.time()
                print(f'Simulation: {index_simulation}. Time of smoothing: {t6-t5} seconds')
            single_simulation = squeeze_first_dimension(single_simulation)
            
            
            add_dict_to_hdf5(self.path_to_constructed_data, index_simulation, single_simulation[index_simulation], path='')
                
        #check shape of each simulation 
        with h5py.File(self.path_to_constructed_data, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                for shape in f[key].keys():
                    print(f"Simulation {key}: shape {shape}, {np.shape(f[key][shape])}")
                    
        #get normalization statistics
        if purpose_of_data == 'training': 
            t7 = time.time()
            self.maxima_or_mean, self.minima_or_std = get_normalization_statistics_progressively(self.path_to_constructed_data, self.which_normalization, self.percentile_high_norm_min_max, self.percentile_low_norm_min_max) 
            t8 = time.time()
            print(f'get normalizations statistics: {t8-t7} seconds')
            
            for key in self.maxima_or_mean:
                print(f'maxima_or_mean {key}, ',self.maxima_or_mean[key])
                
            print(' ')
            
            for key in self.minima_or_std:
                print(f'minima_or_std {key}, ',self.minima_or_std[key])
            
            self.maxima_or_mean = {k: tc.tensor(v, dtype=tc.float32) for k, v in self.maxima_or_mean.items()}
            self.minima_or_std = {k: tc.tensor(v, dtype=tc.float32) for k, v in self.minima_or_std.items()}
            
            #save normalization statistics for training and testing
            with open(f"{self.where_to_save_data}/maxima_or_mean{self.indeces_training_boundaries}.pkl", 'wb') as f:
                pickle.dump(self.maxima_or_mean, f)

            with open(f"{self.where_to_save_data}/minima_or_std{self.indeces_training_boundaries}.pkl", 'wb') as f:
                pickle.dump(self.minima_or_std, f)
                
        elif purpose_of_data == 'validation' or purpose_of_data == 'testing':
            
            with open(f"{self.where_to_save_data}/maxima_or_mean{self.indeces_training_boundaries}.pkl", 'rb') as file:
                self.maxima_or_mean = pickle.load(file)
            
            with open(f"{self.where_to_save_data}/minima_or_std{self.indeces_training_boundaries}.pkl", 'rb') as file:
                self.minima_or_std = pickle.load(file)
                
            for key in self.maxima_or_mean:
                print(f'maxima_or_mean {key}, ',self.maxima_or_mean[key])
                
            print(' ')
            
            for key in self.minima_or_std:
                print(f'minima_or_std {key}, ',self.minima_or_std[key])
                
        #normalize dictionary_per_simulation 
        t9 = time.time()
        with h5py.File(self.path_to_constructed_data, 'r+') as f:
            keys = list(f.keys())
            for key in keys:
                t0 = time.time()
                normalized_dict = self.normalize_dictionary_per_simulation(f[key], self.maxima_or_mean, self.minima_or_std)
                for shape in normalized_dict:
                    f[key][shape][()] = normalized_dict[shape].cpu().numpy()
                t1 = time.time()
                print(f"Time to normalize simulation {key}: {t1-t0}")
                
                for shape in f[key]:
                    print(f"Simulation {key}. Shape {shape}: {np.shape(f[key][shape])})")
        
        #reshape dictionary   
        t11 = time.time()
        
        with h5py.File(self.path_to_constructed_data, 'r+') as f:
            keys = list(f.keys())
            for count, key in enumerate(keys):
                t0 = time.time()
                reshaped_dict = self.reshape_dataset(f[key])
                for shape in reshaped_dict:
                    #delete because shape is different
                    if shape in f[key]:
                        del f[key][shape]
                    f[key].create_dataset(shape, data=reshaped_dict[shape].cpu().numpy(), dtype='float32')
                if self.testing:
                    reshaped_dict['Time'] = extract_time_of_simulation(self.path_to_hdf5, key,subsampling_indeces[count] )
                    op_acts = self.get_operator_actions(key)
                    reshaped_dict['Operator_actions'] = op_acts
                    f[key].create_dataset('Time', data=reshaped_dict['Time'], dtype='float32')
                    actions_group = f[key].create_group('Operator_actions')
                    for action_key, action_value in reshaped_dict['Operator_actions'].items():
                        actions_group.create_dataset(action_key, data=action_value, dtype='float32')
                t1 = time.time()
                
                print(f"Time to reshape simulation {key}: {t1-t0}")
                
                for shape in f[key]:
                    print(f"Simulation {key}. Shape {shape}: {np.shape(f[key][shape])})")
                    
        t12 = time.time()
        print(f'reshape dictionary_per_simulation : {t12-t11} seconds')
        
        #check if there are nan values in the dictionary before saving
        with h5py.File(self.path_to_constructed_data, 'r') as f:
            simulations = list(f.keys())
            for simulation in simulations:
                for shape in f[simulation]:
                    if shape != 'Operator_actions' and shape != 'Time':
                        if np.isnan(f[simulation][shape]).any() or (~np.isfinite(f[simulation][shape])).any():
                            raise TypeError(f"There are still NaN values in final data, check please in simulation {simulation}, shape {shape}")
                    elif shape == 'Time':
                        if np.isnan(f[simulation][shape]).any() or (~np.isfinite(f[simulation][shape])).any():
                            raise TypeError(f"There are still NaN values in final data, check please in simulation {simulation}, shape {shape}")
        #check shapes
        print('')
        print('CHECK SHAPES PLEASE!')  
        print('')
        with h5py.File(self.path_to_constructed_data, 'r') as f:
            simulations = list(f.keys())
            for simulation in simulations:
                for shape in f[simulation]:
                    if shape != 'Time' and shape != 'Operator_actions':
                        size = np.shape(f[simulation][shape])
                    elif shape == 'Time':
                        size = np.shape(f[simulation][shape])
                    else:
                        continue
                    print(f"trajectory {simulation}, shape {shape}: {size} ")
        print(f'Skipped simulations: {skipped_simulations}')
                        
        
            
            
    def make_channels_for_dictionary_per_simulation(self, dict: dict):
        for n_o_s in dict:  # n_o_s is number of simulation
            for m_t in dict[n_o_s]:  # m_t is the mesh type (so 1, 76, 32 and so on)
                field_dict = dict[n_o_s][m_t]
                fields = list(field_dict.keys())
                
                # Get first field to determine structure
                first_field = field_dict[fields[0]]
                size_field = first_field.ndim  # More efficient than len(np.shape())
                
                if size_field == 2:
                    stacked = np.stack([field_dict[field] for field in fields], axis=0)
                    concatenated_array = stacked[None, :, :, :].transpose(0, 2, 1, 3)
                    
                elif size_field == 1:
                    stacked = np.stack([field_dict[field] for field in fields], axis=0)
                    concatenated_array = stacked[None, :, :].transpose(0, 2, 1)
                    
                else:
                    raise TypeError("Something is wrong with data structure")
                dict[n_o_s][m_t] = concatenated_array
        
    
        for i in dict:
            bc_arrays = [
                dict[i]['VDO'],
                dict[i]['UPP_V001'],
            ]
            concatenated_bc = np.concatenate(bc_arrays, axis=-1)
            dict[i]['boundary_conditions_and_time'] = concatenated_bc
            
            for key in ['VDO', 'UPP_V001']:
                dict[i].pop(key)
            self.length_boundaries = np.shape(concatenated_bc[0][-1])
        return dict
    
    

    def tv_smooth_axis1(self, data, weight=0.1, n_jobs=-1):
        """Apply TV denoising along axis=1 for any shape (N, T, ...), parallelized."""
        indices = list(np.ndindex(data.shape[0], *data.shape[2:]))

        def smooth_one(idx, sliced):
            return idx, denoise_tv_chambolle(sliced, weight=weight)

        out = np.empty_like(data, dtype=float)
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(smooth_one)(idx, data[(idx[0], slice(None)) + idx[1:]])
            for idx in indices
        )

        for idx, smoothed in results:
            slc = (idx[0], slice(None)) + idx[1:]
            out[slc] = smoothed

        return out

    def use_smoothing(self, simulation: dict, index_simulation: int):
        keys = simulation[index_simulation].keys()
        found = False
        for i in keys:
            if i == 'boundary_conditions_and_time':
                found = True
                data = simulation[index_simulation][i]  # shape (1, 25184, 26)
                filtered = self.smooth_axis1(data[..., :24])
                simulation[index_simulation][i] = np.concatenate([filtered, data[..., 24:]], axis=-1)
            else:
                simulation[index_simulation][i] = self.smooth_axis1(simulation[index_simulation][i])
        if not found:
            raise TypeError("boundary_conditions_and_time NOT FOUND")
        return simulation

    def smooth_axis1(self, data, n_jobs=-1):
        """Apply smoothing along axis=1, either TV Chambolle or moving average."""
        if self.smoothing_method == 'tv_chambolle':
            return self.tv_smooth_axis1(data, weight=self.weight_denoise_tv_chambolle, n_jobs=n_jobs)
        elif self.smoothing_method == 'moving_average':
            return self.ma_smooth_axis1(data, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

    def ma_smooth_axis1(self, data, n_jobs=-1):
        """Apply moving average smoothing along axis=1, with reflect padding to avoid edge effects."""
        from scipy.ndimage import uniform_filter1d
        
        indices = list(np.ndindex(data.shape[0], *data.shape[2:]))
        out = np.empty_like(data, dtype=float)
        
        def smooth_one(idx, sliced):
            return idx, uniform_filter1d(sliced, size=self.window_length_smoothing, mode='reflect')
        
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(smooth_one)(idx, data[(idx[0], slice(None)) + idx[1:]])
            for idx in indices
        )
        for idx, smoothed in results:
            out[(idx[0], slice(None)) + idx[1:]] = smoothed
        return out
        
    def make_dictionary_unified(self):
        numbers_of_simulation = list(self.dictionary_per_simulation.keys())
        dictionary_unified = {}
        variables = self.dictionary_per_simulation[numbers_of_simulation[0]].keys()
        
        for variable in variables:
            dictionary_unified[variable] = []
            
        for numbers_of_simulation in numbers_of_simulation:
            sim_data = self.dictionary_per_simulation[numbers_of_simulation]
            for variable in variables:
                field = sim_data[variable]
                size = field.shape
                reshaped_field = field.reshape((size[0] * size[1],)+size[2:])
                dictionary_unified[variable].append(reshaped_field)
        for variable in variables:
            dictionary_unified[variable] = np.concatenate(dictionary_unified[variable], axis=0)
        return dictionary_unified
    
    def substitute_NaN_with_zeros(self, dictionary:dict):
        for trajectory in dictionary:
            for shape in dictionary[trajectory]:
                arr = dictionary[trajectory][shape]
                if np.isnan(arr).any():
                    arr[np.isnan(arr)] = 0.0
        return dictionary    
    
    def normalize_dictionary_per_simulation(self, simulation:dict, maxima_or_mean : dict, minima_or_std: dict):
        normalized_dict = {}
        for variable in simulation:
            normalized_field = normalize_fields(np.array(simulation[variable]), maxima_or_mean, minima_or_std, self.which_normalization, self.device)
            normalized_dict[variable] = normalized_field
        return normalized_dict
        
    def reshape_dataset(self, simulation):
        reshaped_dict = {}
        variables = simulation.keys()
        for variable in variables:
            original_data = tc.tensor(np.array(simulation[variable]), device = self.device)
            shape = original_data.size() 
            if shape[-1] == 140:
                original_data = make_faces_array(original_data, self.device)
            elif shape[-1] == 36:
                original_data = tc.reshape(original_data, (shape[0],shape[1],12,3)).cpu()
            elif shape[-1] == 76:
                lower_plenum = original_data[:,:,0]
                mesh = original_data[:,:,1:].reshape(shape[0],shape[1],15,5)
                original_data = mesh
                reshaped_dict['lower_plenum'] = lower_plenum
            elif shape[-1] == 7 or shape[-1] == 13 and len(shape) !=1:
                original_data = tc.reshape(original_data, (shape[0],shape[1]))
            reshaped_dict[variable] = original_data.cpu()
            
        return reshaped_dict
                
    
    
    def get_operator_actions(self, trajectory):
        operator_actions_dict = {}
        with h5py.File(self.path_to_hdf5+'/'+str(trajectory)+'.h5', 'r') as f:
            operator_names = ['t_fbseb', 't1_srv', 'opensrv', 't2_srv', 'tendssg2', 'tpesp','tpessg', 'tcss', 'p_u5', 'tsg2tr']
            for op in operator_names:
                operator_actions_dict[op] = (f['other/private/'+ op][0])/ 3600.0
                if np.isnan(f['other/private/'+ op][0]).any() or not np.isfinite(f['other/private/'+ op][0]).any():
                    raise TypeError(f"Operator action {op} in simulation {trajectory} is NaN")
        return operator_actions_dict
