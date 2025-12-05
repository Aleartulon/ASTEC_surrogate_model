from src.dataset_generation.sliced_dataset.support_functions import *
import h5py
import numpy as np
import torch as tc
import pickle
import time
import gc


class Sliced_Dataset():
    def __init__(self , config_dataset: dict):
        
        self.path_to_dataset = config_dataset['path_to_dataset']
        self.where_to_save_data = config_dataset['where_to_save_data']
        self.t_W = config_dataset['t_W']
        self.device = config_dataset['device']
        self.indeces_training_boundaries = '_'
        self.indeces_validation_boundaries = '_'
        
        for i in config_dataset['indeces_training_boundaries']:
            self.indeces_training_boundaries += str(i) + '_'
        self.indeces_training_boundaries = self.indeces_training_boundaries[:-1]
        
        for i in config_dataset['indeces_validation_boundaries']:
            self.indeces_validation_boundaries += str(i) + '_'
        self.indeces_validation_boundaries = self.indeces_validation_boundaries[:-1]
        
        
    def build_sliced_training_dataset(self, purpose_of_data):
        self.purpose_of_data = purpose_of_data
        
        if self.purpose_of_data == 'training':
            self.path_saved_dataset = f'{self.where_to_save_data}/data_training_normalized_t_W_{self.t_W}{self.indeces_training_boundaries}.h5'
        else:
            self.path_saved_dataset = f'{self.where_to_save_data}/data_validation_normalized_t_W_{self.t_W}{self.indeces_validation_boundaries}.h5'
        
        if self.purpose_of_data  == 'training':
            name_file = f'data_training{self.indeces_training_boundaries}.h5'
        elif self.purpose_of_data == 'validation':
            name_file = f'data_validation{self.indeces_validation_boundaries}.h5'
            
        # open dataset
        with h5py.File(self.path_to_dataset + name_file, 'r') as f:
            #divide datasets in time windows of length t_W, pad and mix the trajectories
            t5 = time.time()
            self.make_dictionary_of_sliced_windows(f)
            t6 = time.time()
            print(f'divide datasets in time windows of length t_W, pad and mix the trajectories: {t6-t5} seconds')
            print('')
            print('CHECK SHAPES PLEASE!')  
            print('')
            with h5py.File(self.path_saved_dataset, 'r') as f:
                for shape in f:
                    print(f"Shape of {shape}, {np.shape(f[shape])}")
        return 0
    
            
    def make_dictionary_of_sliced_windows(self, dictionary_of_simulations:dict):
        numbers_of_simulation = list(dictionary_of_simulations.keys())
        variables = list(dictionary_of_simulations[numbers_of_simulation[0]].keys())
            
        # First pass: calculate total sizes for each variable
        total_windows = {}
        for variable in variables:
            total_windows[variable] = 0
        total_windows['length_of_padding'] = 0
        
        for number_of_simulation in numbers_of_simulation:
            size = np.shape(dictionary_of_simulations[number_of_simulation][variables[0]])
            pad_to_be_added = (self.t_W - (size[0] % self.t_W)) % self.t_W
            num_windows = int((size[0] + pad_to_be_added) / self.t_W)
            
            for variable in variables:
                total_windows[variable] += num_windows
            total_windows['length_of_padding'] += num_windows
            
        #create precomputed files
        with h5py.File(self.path_saved_dataset, 'w') as f_out:
            # Get shape template from first simulation
            first_sim = numbers_of_simulation[0]
            for variable in variables:
                shape = np.shape(dictionary_of_simulations[first_sim][variable])
                dtype = dictionary_of_simulations[first_sim][variable].dtype
                
                # Create dataset with full size
                full_shape = (total_windows[variable], self.t_W) + shape[1:]
                f_out.create_dataset(variable, shape=full_shape, dtype=dtype)

            # Create padding tracker dataset
            f_out.create_dataset('length_of_padding', shape=(total_windows['length_of_padding'], 1), dtype=np.float32)
            
            # Second pass: write data incrementally
            write_idx = {var: 0 for var in variables}
            write_idx['length_of_padding'] = 0
            
            for number_of_simulation in numbers_of_simulation:
                size = np.shape(dictionary_of_simulations[number_of_simulation][variables[0]])
                pad_to_be_added = (self.t_W - (size[0] % self.t_W)) % self.t_W
                num_windows = int((size[0] + pad_to_be_added) / self.t_W)
                
                for variable in variables:
                    field = dictionary_of_simulations[number_of_simulation][variable][()]
                    size = np.shape(field)
                    
                    if pad_to_be_added > 0:
                        padded_tensor = tc.nn.functional.pad(
                            tc.from_numpy(field), 
                            (0, 0) * (len(size) - 1) + (0, pad_to_be_added)
                        )
                        reshaped = tc.reshape(
                            padded_tensor, 
                            (num_windows, self.t_W) + size[1:]
                        )
                    else:
                        reshaped = tc.reshape(
                            tc.from_numpy(field), 
                            (num_windows, self.t_W) + size[1:]
                        )
                    
                    # Write directly to HDF5
                    f_out[variable][write_idx[variable]:write_idx[variable] + num_windows] = \
                        reshaped.cpu().numpy()
                    write_idx[variable] += num_windows
                    
                    # Free memory immediately
                    del field, reshaped
                # Write padding tracker
                padding_tracker = np.zeros((num_windows, 1))
                padding_tracker[-1, 0] = pad_to_be_added
                f_out['length_of_padding'][write_idx['length_of_padding']:
                                        write_idx['length_of_padding'] + num_windows] = padding_tracker
                write_idx['length_of_padding'] += num_windows
                
                # Clear memory
                gc.collect()
                if tc.cuda.is_available():
                    tc.cuda.empty_cache()
        return 0
        
    