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
        self.indeces_training_boundaries = config_dataset['indeces_training_boundaries']
        self.indeces_validation_boundaries = config_dataset['indeces_validation_boundaries']
        
    def build_sliced_training_dataset(self, purpose_of_data):
        self.purpose_of_data = purpose_of_data
        
        if self.purpose_of_data  == 'training':
            name_file = f'data_training_{self.indeces_training_boundaries[0]}_{self.indeces_training_boundaries[1]}.h5'
        elif self.purpose_of_data == 'validation':
            name_file = f'data_validation_{self.indeces_validation_boundaries[0]}_{self.indeces_validation_boundaries[1]}.h5'
            
        # open dataset
        with h5py.File(self.where_to_save_data + name_file, 'r') as f:
            #divide datasets in time windows of length t_W, pad and mix the trajectories
            t5 = time.time()
            self.dictionary_of_sliced_windows = self.make_dictionary_of_sliced_windows(f)
            t6 = time.time()
            
            print(f'divide datasets in time windows of length t_W, pad and mix the trajectories: {t6-t5} seconds')
            print('')
            print('CHECK SHAPES PLEASE!')  
            print('')
            for key in self.dictionary_of_sliced_windows:
                print(key, self.dictionary_of_sliced_windows[key].size())
                
            #save dictionary with sliced windows
            if self.purpose_of_data == 'training':
                with h5py.File(f'{self.where_to_save_data}/data_training_normalized_t_W_{self.t_W}_{self.indeces_training_boundaries[0]}_{self.indeces_training_boundaries[1]}.h5', 'w') as f:
                    dict_to_hdf5(self.dictionary_of_sliced_windows, f)
            elif self.purpose_of_data == 'validation':
                with h5py.File(f'{self.where_to_save_data}/data_validation_normalized_t_W_{self.t_W}_{self.indeces_validation_boundaries[0]}_{self.indeces_validation_boundaries[1]}.h5', 'w') as f:
                    dict_to_hdf5(self.dictionary_of_sliced_windows, f)
            
            # Clear GPU memory immediately after saving
            del self.dictionary_of_sliced_windows
            
            
            gc.collect()
            tc.cuda.empty_cache()
        
            print(f"GPU memory after build_{purpose_of_data}: {tc.cuda.memory_allocated()/1e9:.2f} GB")
            
        return 0
    
            
    def make_dictionary_of_sliced_windows(self, dictionary_of_simulations:dict):
        numbers_of_simulation = list(dictionary_of_simulations.keys())
        dictionary_of_sliced_windows = {}
        variables = dictionary_of_simulations[numbers_of_simulation[0]].keys()
        for variable in variables:
            size = (1, self.t_W) + np.shape(dictionary_of_simulations[numbers_of_simulation[0]][variable])[2:] 
            dictionary_of_sliced_windows[variable] = []
        dictionary_of_sliced_windows['length_of_padding'] = []
        
        for number_of_simulation in numbers_of_simulation:
            for variable in variables:
                field = dictionary_of_simulations[number_of_simulation][variable][()]
                size = np.shape(dictionary_of_simulations[number_of_simulation][variable])
                pad_to_be_added = (self.t_W - (size[0]%self.t_W)) % self.t_W
                if pad_to_be_added > 0:
                    padded_tensor = tc.nn.functional.pad(tc.from_numpy(field), (0, 0)*(len(size)-1) + (0, pad_to_be_added))
                    padded_tensor = tc.reshape(padded_tensor, (int((size[0] +pad_to_be_added)/self.t_W), self.t_W) + size[1:])
                    dictionary_of_sliced_windows[variable].append(padded_tensor)
                else:
                   dictionary_of_sliced_windows[variable].append(tc.reshape(tc.from_numpy(field), ((int(size[0] /self.t_W)), self.t_W) + size[1:]))
            padding_tracker = tc.zeros(int((size[0]+pad_to_be_added)/self.t_W))
            padding_tracker[-1] = pad_to_be_added
            dictionary_of_sliced_windows['length_of_padding'].append(padding_tracker)
        dictionary_of_sliced_windows['length_of_padding'] = tc.cat(dictionary_of_sliced_windows['length_of_padding'], dim=0).unsqueeze(-1)
        for variable in variables:
            dictionary_of_sliced_windows[variable] = tc.cat(dictionary_of_sliced_windows[variable], dim=0)
        return dictionary_of_sliced_windows
    