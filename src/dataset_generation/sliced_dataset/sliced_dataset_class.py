from src.dataset_generation.sliced_dataset.support_functions import *
import h5py
import numpy as np
import torch as tc
import pickle
import time
import gc


class Sliced_Dataset():
    def __init__(self , config_dataset: dict):
        
        self.path_to_hdf5 = config_dataset['path_to_hdf5']
        self.where_to_save_data = config_dataset['where_to_save_data']
        self.t_W = config_dataset['t_W']
        self.device = config_dataset['device']
        
    def build_training_dataset(self, indeces, purpose_of_data):
        self.purpose_of_data = purpose_of_data
        t1 = time.time()
        self.dictionary_per_simulation, _ = extract_input_output_bc_variables(self.path_to_hdf5, indeces) #build dictionary of data divided by number of simulation
        t2 = time.time()
        print(f'Build dictionary of data divided by number of simulation: {t2-t1} seconds')
        self.dictionary_per_simulation = self.make_channels_for_dictionary_per_simulation(self.dictionary_per_simulation) #build dictionary of data divided by simulations and make channels per spatial domain
        t3 = time.time()
        print(f'build dictionary of data divided by simulations and make channels per spatial domain: {t3-t2} seconds')
        self.dictionary_per_simulation = self.substitute_NaN_with_zeros(self.dictionary_per_simulation) #substitute with zeros the NaN values
        t4 = time.time()
        print(f'substitute with zeros the NaN values: {t4-t3} seconds')
        
        #save dictionary_per_simulation to hdf5s if self.save_dictionary_per_time_lengths is true
        if self.save_dictionary_per_time_lengths:
            with h5py.File(self.where_to_save_data+'/data_'+self.purpose_of_data+'.h5', 'w') as f:
                dict_to_hdf5(self.dictionary_per_simulation, f) 
        
        #get normalization statistics
        if purpose_of_data == 'training': 
            # get unified dataset to get the normalizations statistics
            dictionary_unified = self.make_dictionary_unified()
            self.maxima_or_mean, self.minima_or_std = get_normalization_statistics(dictionary_unified, self.which_normalization) 
            
            for key in self.maxima_or_mean:
                print(f'maxima_or_mean {key}, ',self.maxima_or_mean[key])
                
            print(' ')
            
            for key in self.minima_or_std:
                print(f'minima_or_std {key}, ',self.minima_or_std[key])
            
            self.maxima_or_mean = {k: tc.tensor(v, dtype=tc.float32) for k, v in self.maxima_or_mean.items()}
            self.minima_or_std = {k: tc.tensor(v, dtype=tc.float32) for k, v in self.minima_or_std.items()}
            
            #save normalization statistics for training and testing
            with open(self.where_to_save_data+'/maxima_or_mean.pkl', 'wb') as f:
                pickle.dump(self.maxima_or_mean, f)

            with open(self.where_to_save_data+'/minima_or_std.pkl', 'wb') as f:
                pickle.dump(self.minima_or_std, f)
                
        elif purpose_of_data == 'validation':
            
            with open(self.where_to_save_data + '/maxima_or_mean.pkl', 'rb') as file:
                self.maxima_or_mean = pickle.load(file)
            
            with open(self.where_to_save_data + '/minima_or_std.pkl', 'rb') as file:
                self.minima_or_std = pickle.load(file)
            
        #divide datasets in time windows of length t_W, pad and mix the trajectories
        t5 = time.time()
        self.dictionary_of_sliced_windows = self.make_dictionary_of_sliced_windows()
        t6 = time.time()
        print(f'divide datasets in time windows of length t_W, pad and mix the trajectories: {t6-t5} seconds')
        #normalize dictionary with sliced windows and reshape in correct shapes
        self.dictionary_of_sliced_windows = self.normalize_dictionary_of_sliced_windows()
        t7 = time.time()
        print(f'normalize dictionary with sliced windows and reshape in correct shapes: {t7-t6} seconds')
        #check if there are nan values in the dictionary before saving
        for key in self.dictionary_of_sliced_windows:
            if tc.isnan(self.dictionary_of_sliced_windows[key]).any() or (~tc.isfinite(self.dictionary_of_sliced_windows[key])).any():
                raise TypeError(f"There are still NaN values in final data, check please in {key}")
        #check shapes
        print('')
        print('CHECK SHAPES PLEASE!')  
        print('')
        for key in self.dictionary_of_sliced_windows:
            print(key, self.dictionary_of_sliced_windows[key].size())
        #save normalized dictionary with sliced windows
        with h5py.File(f'{self.where_to_save_data}/data_{self.purpose_of_data}_normalized_t_W_'+str(self.t_W)+'.h5', 'w') as f:
            dict_to_hdf5(self.dictionary_of_sliced_windows, f)
        
        # Clear GPU memory immediately after saving
        del self.dictionary_per_simulation
        del self.dictionary_of_sliced_windows
        
        
        gc.collect()
        tc.cuda.empty_cache()
    
        print(f"GPU memory after build_{purpose_of_data}: {tc.cuda.memory_allocated()/1e9:.2f} GB")
            
        return 0
    
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
                dict[i]['primary_to_vessel'],
                dict[i]['VDO'],
                dict[i]['UPP_V001'],
                dict[i]['vessel_to_primary']
            ]
            dict[i]['boundary_conditions_and_time'] = np.concatenate(bc_arrays, axis=-1)
            
            for key in ['primary_to_vessel', 'vessel_to_primary', 'VDO', 'UPP_V001']:
                dict[i].pop(key)
    
        return dict
    
    def make_dictionary_unified(self):
        numbers_of_simulation = list(self.dictionary_per_simulation.keys())
        dictionary_unified = {}
        variables = self.dictionary_per_simulation[numbers_of_simulation[0]].keys()
        for numbers_of_simulation in numbers_of_simulation:
            for variable in variables:
                size = np.shape(self.dictionary_per_simulation[numbers_of_simulation][variable])
                reshaped_field = np.reshape(self.dictionary_per_simulation[numbers_of_simulation][variable], (size[0] * size[1],)+size[2:])
                dictionary_unified[variable] = reshaped_field
                
        return dictionary_unified
            
    def make_dictionary_of_sliced_windows(self):
        numbers_of_simulation = list(self.dictionary_per_simulation.keys())
        dictionary_of_sliced_windows = {}
        variables = self.dictionary_per_simulation[numbers_of_simulation[0]].keys()
        for variable in variables:
            size = (1, self.t_W) + np.shape(self.dictionary_per_simulation[numbers_of_simulation[0]][variable])[2:] 
            dictionary_of_sliced_windows[variable] = []
        dictionary_of_sliced_windows['length_of_padding'] = []
        
        for number_of_simulation in numbers_of_simulation:
            for variable in variables:
                size = np.shape(self.dictionary_per_simulation[number_of_simulation][variable])
                pad_to_be_added = (self.t_W - (size[1]%self.t_W)) % self.t_W
                if pad_to_be_added > 0:
                    padded_tensor = tc.nn.functional.pad(tc.from_numpy(self.dictionary_per_simulation[number_of_simulation][variable]), (0, 0)*(len(size)-2) + (0, pad_to_be_added))
                    padded_tensor = tc.reshape(padded_tensor, (size[0] * int((size[1]+pad_to_be_added)/self.t_W), self.t_W) + size[2:])
                    dictionary_of_sliced_windows[variable].append(padded_tensor)
                else:
                   dictionary_of_sliced_windows[variable].append(tc.reshape(tc.from_numpy(self.dictionary_per_simulation[number_of_simulation][variable]), (size[0] * int((size[1])/self.t_W), self.t_W) + size[2:]))
            padding_tracker = tc.zeros(int((size[1]+pad_to_be_added)/self.t_W))
            padding_tracker[-1] = pad_to_be_added
            dictionary_of_sliced_windows['length_of_padding'].append(padding_tracker)
        dictionary_of_sliced_windows['length_of_padding'] = tc.cat(dictionary_of_sliced_windows['length_of_padding'], dim=0).unsqueeze(-1)
        for variable in variables:
            dictionary_of_sliced_windows[variable] = tc.cat(dictionary_of_sliced_windows[variable], dim=0)
        return dictionary_of_sliced_windows
    
    def normalize_dictionary_of_sliced_windows(self):
        
        normalized_dict = {}
        variables = self.dictionary_of_sliced_windows.keys()
        for variable in variables:
            if variable != 'length_of_padding':
                print(variable)
                t1 = time.time()
                normalized_data = normalize_fields(self.dictionary_of_sliced_windows[variable], self.maxima_or_mean, self.minima_or_std, self.which_normalization, self.device)
                t2 = time.time()
                print('time to normalize: ',t2-t1)
                shape = np.shape(normalized_data)
                if shape[-1] == 140:
                    normalized_data = make_faces_array(normalized_data, self.device)
                    normalized_dict[variable] = normalized_data.cpu()
                elif shape[-1] == 36:
                    normalized_data = tc.reshape(normalized_data, (shape[0],shape[1],shape[2],12,3))
                    normalized_dict[variable] = normalized_data.cpu()
                elif shape[-1] == 76:
                    lower_plenum = normalized_data[:,:,:,0]
                    mesh = normalized_data[:,:,:,1:].reshape(shape[0],shape[1],shape[2],15,5)
                    normalized_data = mesh
                    normalized_dict[variable] = normalized_data.cpu()
                    normalized_dict['lower_plenum'] = lower_plenum.cpu()
                elif len(shape) == 3:
                    normalized_dict[variable] = normalized_data.cpu()
                else:
                    raise TypeError(f"Something is wrong with data, shape is {shape}")
            else:
                normalized_dict[variable] = self.dictionary_of_sliced_windows[variable].cpu()
            t3 = time.time()
            print('time to reshape: ',t3-t2)
        return normalized_dict
    
    def substitute_NaN_with_zeros(self, dictionary):
        for trajectory in dictionary:
            for shape in dictionary[trajectory]:
                arr = dictionary[trajectory][shape]
                if np.isnan(arr).any():
                    arr[np.isnan(arr)] = 0.0
        return dictionary
                
    def build_testing_dataset(self, indeces):
        
        with open(self.where_to_save_data + '/maxima_or_mean.pkl', 'rb') as file:
            maxima_or_mean = pickle.load(file)
        
        with open(self.where_to_save_data + '/minima_or_std.pkl', 'rb') as file:
            minima_or_std = pickle.load(file)
        
        dictionary_per_trajectory, self.time_of_simulations = extract_input_output_bc_variables(self.path_to_hdf5, indeces) #build dictionary of data divided by numbers of simulations
        dictionary_per_trajectory = self.make_channels_for_dictionary_per_simulation(dictionary_per_trajectory) #build dictionary of data divided by numbers of simulation and make channels per spatial domain
        dictionary_per_trajectory = self.substitute_NaN_with_zeros(dictionary_per_trajectory)
        dictionary_per_trajectory = self.normalize_testing_dataset(dictionary_per_trajectory, maxima_or_mean, minima_or_std) #normalize testing dataset according to training statistics
        dictionary_per_trajectory = self.reshape_testing_dataset(dictionary_per_trajectory) #reshape dictionary
        
        #check if there are nan values in the dictionary before saving
        for number_of_simulation in dictionary_per_trajectory:
            shapes = list(dictionary_per_trajectory[number_of_simulation].keys())
            for shape in shapes:
                if shape != 'Operator_actions' and shape != 'Time':
                    if tc.isnan(dictionary_per_trajectory[number_of_simulation][shape]).any() or (~tc.isfinite(dictionary_per_trajectory[number_of_simulation][shape])).any():
                        raise TypeError(f"There are still NaN values in final data, check please in simulation {number_of_simulation}, shape {shape}")
                elif shape == 'Time':
                    if np.isnan(dictionary_per_trajectory[number_of_simulation][shape]).any() or (~np.isfinite(dictionary_per_trajectory[number_of_simulation][shape])).any():
                        raise TypeError(f"There are still NaN values in final data, check please in simulation {number_of_simulation}, shape {shape}")
            
        #check shapes
        print('')
        print('CHECK SHAPES PLEASE!')  
        print('')
        for number_of_simulation in dictionary_per_trajectory:
            shapes = list(dictionary_per_trajectory[number_of_simulation].keys())
            for shape in shapes:
                print(f"shape {number_of_simulation}, {shape}", np.shape(dictionary_per_trajectory[number_of_simulation][shape]))
            
        #save dictionary_per_simulation to hdf5s if self.save_dictionary_per_time_lengths is true
        with h5py.File(self.where_to_save_data+'/data_testing.h5', 'w') as f:
            dict_to_hdf5(dictionary_per_trajectory, f) 
    
    def normalize_testing_dataset(self, dictionary_per_trajectory:dict, maxima_or_mean : dict, minima_or_std: dict):
        normalized_dict = {}
        for number_of_simulation in dictionary_per_trajectory:
            for variable in dictionary_per_trajectory[number_of_simulation]:
                normalized_field = normalize_fields(tc.tensor(dictionary_per_trajectory[number_of_simulation][variable]), maxima_or_mean, minima_or_std, self.which_normalization, self.device)
                normalized_dict.setdefault(number_of_simulation, {})[variable] = normalized_field
        return normalized_dict
        
    def reshape_testing_dataset(self, dictionary_per_trajectory):
        reshaped_dict = {}
        trajectories = dictionary_per_trajectory.keys()
        for count, trajectory in enumerate(trajectories):
            variables = dictionary_per_trajectory[trajectory].keys()
            for variable in variables:
                original_data = dictionary_per_trajectory[trajectory][variable]
                shape = original_data.size()
                if shape[-1] == 140:
                    original_data = make_faces_array(original_data, self.device)
                elif shape[-1] == 36:
                    original_data = tc.reshape(original_data, (shape[0],shape[1],shape[2],12,3)).cpu()
                elif shape[-1] == 76:
                    lower_plenum = original_data[:,:,:,0]
                    mesh = original_data[:,:,:,1:].reshape(shape[0],shape[1],shape[2],15,5)
                    original_data = mesh
                    reshaped_dict.setdefault(trajectory, {})['lower_plenum'] = lower_plenum
                elif shape[-1] == 7 or shape[-1] == 13 and len(shape) !=2:
                    original_data = tc.reshape(original_data, (shape[0],shape[1],shape[2]))
                reshaped_dict.setdefault(trajectory, {})[variable] = original_data.cpu()
            
            reshaped_dict.setdefault(trajectory, {})['Time'] = self.time_of_simulations[count]
            op_acts = self.get_operator_actions(trajectory)
            reshaped_dict.setdefault(trajectory, {})['Operator_actions'] = op_acts
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
                
