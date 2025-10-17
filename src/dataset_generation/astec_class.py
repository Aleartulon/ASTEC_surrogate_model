from src.dataset_generation.support_functions import *
import h5py
import numpy as np
import torch as tc
import pickle


class Astec_Dataset():
    def __init__(self , config_dataset: dict):
        
        self.path_to_hdf5 = config_dataset['path_to_hdf5']
        self.t_W = config_dataset['t_W']
        self.save_dictionary_per_time_lengths = config_dataset['save_dictionary_per_time_lengths']
        self.which_normalization = config_dataset['which_normalization']
        
    def build_training_dataset(self, indeces, purpose_of_data):
        self.purpose_of_data = purpose_of_data
        self.dictionary_per_simulation, _ = extract_input_output_bc_variables(self.path_to_hdf5, indeces) #build dictionary of data divided by number of simulation
        self.dictionary_per_simulation = self.make_channels_for_dictionary_per_simulation(self.dictionary_per_simulation) #build dictionary of data divided by simulations and make channels per spatial domain
        self.dictionary_per_simulation = self.substitute_NaN_with_zeros(self.dictionary_per_simulation) #substitute with zeros the NaN values
        
        #save dictionary_per_simulation to hdf5s if self.save_dictionary_per_time_lengths is true
        if self.save_dictionary_per_time_lengths:
            with h5py.File(self.path_to_hdf5+'/data_'+self.purpose_of_data+'.h5', 'w') as f:
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
            
            self.maxima_or_mean = {k: np.float32(v) for k, v in self.maxima_or_mean.items()}
            self.minima_or_std = {k: np.float32(v) for k, v in self.minima_or_std.items()}
            
            #save normalization statistics for training and testing
            with open(self.path_to_hdf5+'/maxima_or_mean.pkl', 'wb') as f:
                pickle.dump(self.maxima_or_mean, f)

            with open(self.path_to_hdf5+'/minima_or_std.pkl', 'wb') as f:
                pickle.dump(self.minima_or_std, f)
                
        elif purpose_of_data == 'validation':
            
            with open(self.path_to_hdf5 + '/maxima_or_mean.pkl', 'rb') as file:
                self.maxima_or_mean = pickle.load(file)
            
            with open(self.path_to_hdf5 + '/minima_or_std.pkl', 'rb') as file:
                self.minima_or_std = pickle.load(file)
            
        #divide datasets in time windows of length t_W, pad and mix the trajectories
        self.dictionary_of_sliced_windows = self.make_dictionary_of_sliced_windows()

        #normalize dictionary with sliced windows and reshape in correct shapes
        self.dictionary_of_sliced_windows = self.normalize_dictionary_of_sliced_windows()
        
        #check if there are nan values in the dictionary before saving
        for key in self.dictionary_of_sliced_windows:
            if np.isnan(self.dictionary_of_sliced_windows[key]).any() or not np.isfinite(self.dictionary_of_sliced_windows[key]).any():
                raise TypeError(f"There are still NaN values in final data, check please in {key}")
        #check shapes
        print('')
        print('CHECK SHAPES PLEASE!')  
        print('')
        for key in self.dictionary_of_sliced_windows:
            print(key, np.shape(self.dictionary_of_sliced_windows[key]))
        #save normalized dictionary with sliced windows
        with h5py.File(f'{self.path_to_hdf5}/data_{self.purpose_of_data}_normalized_t_W_500.h5', 'w') as f:
            dict_to_hdf5(self.dictionary_of_sliced_windows, f)
            
        return 0
    
    def make_channels_for_dictionary_per_simulation(self, dict:dict):
        for n_o_s in dict: #n_o_s is number of simulation
            for m_t in dict[n_o_s]: #m_t is the mesh type (so 1, 76, 32 and so on )
                fields = list(dict[n_o_s][m_t].keys())
                size_field = len(np.shape(dict[n_o_s][m_t][fields[0]])) #final [0] to take first batch in case there are multiple trajectories with the same time lenght
                if  size_field == 3:
                    concatenated_array = np.concatenate([np.array(dict[n_o_s][m_t][field])[:,:,None,:] for field in dict[n_o_s][m_t]], axis = 2)
                elif size_field == 2:
                    concatenated_array = np.concatenate([np.array(dict[n_o_s][m_t][field])[:,:, None] for field in dict[n_o_s][m_t]], axis = -1)
                else:
                    raise TypeError("Something is wrong with data structure")   
                 
                dict[n_o_s][m_t] = concatenated_array
        # now mix the bcs
        for i in dict:
            dict[i]['boundary_conditions_and_time'] = np.concatenate([dict[i]['primary_to_vessel'], dict[i]['vessel_to_primary']],axis = -1)
            dict[i].pop('primary_to_vessel')
            dict[i].pop('vessel_to_primary')
        
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
            dictionary_of_sliced_windows[variable] = tc.tensor([])
        dictionary_of_sliced_windows['length_of_padding'] = tc.tensor([])
        
        for number_of_simulation in numbers_of_simulation:
            for variable in variables:
                size = np.shape(self.dictionary_per_simulation[number_of_simulation][variable])
                pad_to_be_added = (self.t_W - (size[1]%self.t_W)) % self.t_W
                padded_tensor = tc.nn.functional.pad(tc.tensor(np.array(self.dictionary_per_simulation[number_of_simulation][variable])), (0, 0)*(len(size)-2) + (0, pad_to_be_added))
                padded_tensor = tc.reshape(padded_tensor, (size[0] * int((size[1]+pad_to_be_added)/self.t_W), self.t_W) + size[2:])
                dictionary_of_sliced_windows[variable] = tc.concatenate((dictionary_of_sliced_windows[variable], padded_tensor), axis = 0)
                padding_tracker = tc.zeros(int((size[1]+pad_to_be_added)/self.t_W))
            padding_tracker[-1] = pad_to_be_added
            dictionary_of_sliced_windows['length_of_padding'] = tc.concatenate((dictionary_of_sliced_windows['length_of_padding'], padding_tracker), axis = 0)
        dictionary_of_sliced_windows['length_of_padding'] = dictionary_of_sliced_windows['length_of_padding'].unsqueeze(-1)
        return dictionary_of_sliced_windows
    
    def normalize_dictionary_of_sliced_windows(self):
        
        normalized_dict = {}
        variables = self.dictionary_of_sliced_windows.keys()
        for variable in variables:
            if variable != 'length_of_padding':
                original_data = self.dictionary_of_sliced_windows[variable][:]
                normalized_data = normalize_fields(original_data, self.maxima_or_mean, self.minima_or_std, self.which_normalization)
                shape = np.shape(normalized_data)
                if shape[-1] == 140:
                    normalized_data = make_faces_array(normalized_data)
                elif shape[-1] == 36:
                    normalized_data = np.reshape(normalized_data, (shape[0],shape[1],shape[2],12,3))
                elif shape[-1] == 76:
                    lower_plenum = normalized_data[:,:,:,0]
                    mesh = normalized_data[:,:,:,1:].reshape(shape[0],shape[1],shape[2],15,5)
                    normalized_data = mesh
                    normalized_dict['lower_plenum'] = lower_plenum
                elif shape[-1] == 7 or shape[-1] == 13 and len(shape) !=2:
                    normalized_data = np.reshape(normalized_data, (shape[0],shape[1],shape[2]))
                normalized_dict[variable] = normalized_data
            else:
                normalized_dict[variable] = self.dictionary_of_sliced_windows[variable]
                
        return normalized_dict
    
    def substitute_NaN_with_zeros(self, dictionary):
        trajectories = list(dictionary.keys())
        shapes = dictionary[trajectories[0]].keys()
        for trajectory in trajectories:
            for shape in shapes:
                dictionary[trajectory][shape] = np.nan_to_num(dictionary[trajectory][shape], nan=0.0)
        return dictionary
                
    def build_testing_dataset(self, indeces):
        
        with open(self.path_to_hdf5 + '/maxima_or_mean.pkl', 'rb') as file:
            maxima_or_mean = pickle.load(file)
        
        with open(self.path_to_hdf5 + '/minima_or_std.pkl', 'rb') as file:
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
                if shape != 'Operator_actions':
                    if np.isnan(dictionary_per_trajectory[number_of_simulation][shape]).any() or not np.isfinite(dictionary_per_trajectory[number_of_simulation][shape]).any():
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
        with h5py.File(self.path_to_hdf5+'/data_testing.h5', 'w') as f:
            dict_to_hdf5(dictionary_per_trajectory, f) 
    
    def normalize_testing_dataset(self, dictionary_per_trajectory:dict, maxima_or_mean : dict, minima_or_std: dict):
        normalized_dict = {}
        for number_of_simulation in dictionary_per_trajectory:
            for variable in dictionary_per_trajectory[number_of_simulation]:
                normalized_field = normalize_fields(tc.tensor(dictionary_per_trajectory[number_of_simulation][variable]), maxima_or_mean, minima_or_std, self.which_normalization)
                normalized_dict.setdefault(number_of_simulation, {})[variable] = normalized_field
        return normalized_dict
        
    def reshape_testing_dataset(self, dictionary_per_trajectory):
        reshaped_dict = {}
        trajectories = dictionary_per_trajectory.keys()
        for count, trajectory in enumerate(trajectories):
            variables = dictionary_per_trajectory[trajectory].keys()
            for variable in variables:
                original_data = dictionary_per_trajectory[trajectory][variable]
                shape = np.shape(original_data)
                if shape[-1] == 140:
                    original_data = make_faces_array(original_data)
                elif shape[-1] == 36:
                    original_data = np.reshape(original_data, (shape[0],shape[1],shape[2],12,3))
                elif shape[-1] == 76:
                    lower_plenum = original_data[:,:,:,0]
                    mesh = original_data[:,:,:,1:].reshape(shape[0],shape[1],shape[2],15,5)
                    original_data = mesh
                    reshaped_dict.setdefault(trajectory, {})['lower_plenum'] = lower_plenum
                elif shape[-1] == 7 or shape[-1] == 13 and len(shape) !=2:
                    original_data = np.reshape(original_data, (shape[0],shape[1],shape[2]))
                reshaped_dict.setdefault(trajectory, {})[variable] = original_data
            
            reshaped_dict.setdefault(trajectory, {})['Time'] = self.time_of_simulations[count]
            op_acts = self.get_operator_actions(trajectory)
            reshaped_dict.setdefault(trajectory, {})['Operator_actions'] = op_acts
        return reshaped_dict
    
    def get_operator_actions(self, trajectory):
        operator_actions_dict = {}
        with h5py.File(self.path_to_hdf5+'/'+str(trajectory)+'.h5', 'r') as f:
            operator_names = ['t_fbseb', 't1_srv', 'opensrv', 't2_srv', 'tendssg2', 'tpesp','tpessg', 'tcss', 'p_u5', 'tsg2tr']
            for op in operator_names:
                operator_actions_dict[op] = f['other/private/'+ op][0]
                if np.isnan(f['other/private/'+ op][0]).any() or not np.isfinite(f['other/private/'+ op][0]).any():
                    raise TypeError(f"Operator action {op} in simulation {trajectory} is NaN")
        return operator_actions_dict
                
