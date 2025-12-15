import numpy as np
import torch as tc
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from src.models.AE_NODE.training.architecture import *
from torch.utils.data import Dataset, DataLoader, get_worker_info
import subprocess

def build_dataset(batch_size:int, time_window: int, data_training_path: str, data_validation_path:str, 
                  number_of_workers:int, path_to_data: str, where_to_save:str , 
                  which_normalization:str, device :tc.device, training_boundaries:list, 
                  validation_boundaries:list, all_on_gpu:bool, pin_memory: bool, indeces_training_boundaries:str, indeces_validation_boundaries:str):
    
    training_path = f"{data_training_path}{str(time_window)}{indeces_training_boundaries}.h5"
    validation_path = f"{data_validation_path}{str(time_window)}{indeces_validation_boundaries}.h5"
    #build dataset made out of 'time_window' chunks
    subprocess.run(['python', '-m', 'src.dataset_generation.sliced_dataset.main', 
                '--t_W', str(time_window), 
                '--path_to_dataset', path_to_data, 
                '--where_to_save_data', where_to_save, 
                '--device', device, 
                '--indeces_training_boundaries', ' ,'.join(map(str, training_boundaries)),
                '--indeces_validation_boundaries', ' ,'.join(map(str, validation_boundaries))])
    tc.cuda.empty_cache()
    # build dataset and dataloader
    dataset_training = ASTEC_Dataset(training_path, all_on_gpu, device)
    dataset_validation = ASTEC_Dataset(validation_path, all_on_gpu, device)
    length_dataset = dataset_training.size
    if batch_size > length_dataset:
        batch_size = max(1, length_dataset // 10)
    print('-------------------------------------------')
    print('Length dataset: ', length_dataset)
    print('Batch size: ', batch_size)
    print('-------------------------------------------')
    training_loader = DataLoader(dataset_training, batch_size = batch_size, num_workers = number_of_workers, shuffle=True,drop_last=False,pin_memory=pin_memory, prefetch_factor=2 if number_of_workers > 0 else None)
    validation_loader = DataLoader(dataset_validation, batch_size = batch_size, num_workers = number_of_workers, shuffle=True,drop_last=False,pin_memory=pin_memory)
    
    return training_loader, validation_loader
  
def save_checkpoint(encoder, f , decoder, optimizer, scheduler, epoch, loss_value, loss_coefficients_AR, before_next_window_change, how_many_datasets_creations, autoregressive_step, full_training_count,filepath):
  
    checkpoint = {
            'encoder':encoder.state_dict(),
            'f':f.state_dict(),
            'decoder':decoder.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch' : epoch,
            'loss_value' : loss_value,
            'loss_coefficients_AR' : loss_coefficients_AR,
            'before_next_window_change' : before_next_window_change,
            'how_many_datasets_creations' : how_many_datasets_creations,
            'autoregressive_step': autoregressive_step,
            'full_training_count': full_training_count,
        }
    tc.save(checkpoint, filepath)


def load_checkpoint(encoder, f , decoder, optimizer, scheduler, filepath, device):

    checkpoint = tc.load(filepath, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    f.load_state_dict(checkpoint['f'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss_value = checkpoint['loss_value']
    loss_coefficients_AR = checkpoint['loss_coefficients_AR']
    before_next_window_change = checkpoint['before_next_window_change']
    how_many_datasets_creations = checkpoint['how_many_datasets_creations']
    autoregressive_step = checkpoint['autoregressive_step']
    full_training_count = checkpoint['full_training_count']
        
    return encoder, f , decoder, optimizer, scheduler , epoch, loss_value, loss_coefficients_AR, before_next_window_change, how_many_datasets_creations, autoregressive_step, full_training_count


class ASTEC_Dataset(Dataset):
    def __init__(self, path:str, all_on_gpu:bool, device:tc.device):
        self.path = path
        self.all_on_gpu = all_on_gpu
        if not all_on_gpu:
            with h5py.File(self.path, 'r') as f:
                self.size = f['dictionary_of_input_variables_1'].shape[0]
        else:
            # Load all data into GPU memory at initialization
            with h5py.File(path, 'r') as f:
                self.dict_vars_1 = tc.from_numpy(f['dictionary_of_input_variables_1'][:]).float().to(device)
                self.dict_vars_36 = tc.from_numpy(f['dictionary_of_input_variables_36'][:]).float().to(device)
                self.dict_vars_76 = tc.from_numpy(f['dictionary_of_input_variables_76'][:]).float().to(device)
                self.lower_plenum = tc.from_numpy(f['lower_plenum'][:]).float().to(device)
                self.dict_vars_140 = tc.from_numpy(f['dictionary_of_input_variables_140'][:]).float().to(device)
                
                bc_data = tc.from_numpy(f['boundary_conditions_and_time'][:]).float().to(device)
                self.boundary_conditions = bc_data[:, :, :-2]
                self.time = bc_data[:, :, -2]
                
                self.length_of_padding = tc.from_numpy(f['length_of_padding'][:]).float().to(device)
                
                self.size = self.dict_vars_1.shape[0]
            
    
    def __getitem__(self, idx):
        if not self.all_on_gpu:
            # Each worker opens its own file handle on first access
            # Use thread/process-local storage
            if not hasattr(self, '_file_handle'):
                self._file_handle = h5py.File(self.path, 'r')
            
            f = self._file_handle
            
            dictionary_of_input_variables_1 = tc.from_numpy(f['dictionary_of_input_variables_1'][idx]).float()
            dictionary_of_input_variables_36 = tc.from_numpy(f['dictionary_of_input_variables_36'][idx]).float()
            dictionary_of_input_variables_76 = tc.from_numpy(f['dictionary_of_input_variables_76'][idx]).float()
            lower_plenum = tc.from_numpy(f['lower_plenum'][idx]).float()
            dictionary_of_input_variables_140 = tc.from_numpy(f['dictionary_of_input_variables_140'][idx]).float()
            
            bc_data = f['boundary_conditions_and_time'][idx]
            boundary_conditions = tc.from_numpy(bc_data[:, :-2]).float()
            time = tc.from_numpy(bc_data[:, -2]).float()
            
            length_of_padding = tc.from_numpy(f['length_of_padding'][idx]).float()
            
            return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, 
                    dictionary_of_input_variables_76, lower_plenum, 
                    dictionary_of_input_variables_140], boundary_conditions, time, length_of_padding
        else:
            return ([self.dict_vars_1[idx], 
                 self.dict_vars_36[idx], 
                 self.dict_vars_76[idx], 
                 self.lower_plenum[idx], 
                 self.dict_vars_140[idx]], 
                self.boundary_conditions[idx], 
                self.time[idx], 
                self.length_of_padding[idx])
    
    def __len__(self):
        return self.size
    
def standard_and_inverse_normalization_field(x: list, maxima_or_mean: dict, minima_or_std:dict, normalization: bool, inverse: bool):
    x_denormalized = []
    
    for _, i in enumerate(x):
        
        if i.size(-1) == 57 and len(i.size()) == 3:
            maximum_or_mean = maxima_or_mean['dictionary_of_input_variables_1'][None,None,:]
            minimum_or_std = minima_or_std['dictionary_of_input_variables_1'][None,None,:]

        elif i.size(-1) == 3:
            maximum_or_mean = maxima_or_mean['dictionary_of_input_variables_36'][None,None,:,None,None]
            minimum_or_std = minima_or_std['dictionary_of_input_variables_36'][None,None,:,None,None]
            
        elif i.size(-1) == 5 and len(i.size()) == 5:
            maximum_or_mean = maxima_or_mean['dictionary_of_input_variables_76'][None,None,:,None,None]
            minimum_or_std = minima_or_std['dictionary_of_input_variables_76'][None,None,:,None,None]
        
        elif i.size(-1) == 18:
            maximum_or_mean = maxima_or_mean['dictionary_of_input_variables_76'][None,None,:]
            minimum_or_std = minima_or_std['dictionary_of_input_variables_76'][None,None,:]
            
        elif i.size(-1) == 9:
            maximum_or_mean = maxima_or_mean['dictionary_of_input_variables_140'][None,None,:,None,None]
            minimum_or_std = minima_or_std['dictionary_of_input_variables_140'][None,None,:,None,None]
            
        elif i.size(-1) == 6:
            maximum_or_mean = maxima_or_mean['boundary_conditions_and_time'][None,None,:-1]
            minimum_or_std = minima_or_std['boundary_conditions_and_time'][None,None,:-1]
            
        else:
            raise TypeError(f"Something is wrong with data structure, size is {i.size()}")  
    
        if normalization == 'min_max':
            if inverse:
                denorm = (i * (maximum_or_mean - minimum_or_std) + minimum_or_std)
                x_denormalized.append(denorm)
            else:
                norm = ((i - minimum_or_std)/(maximum_or_mean - minimum_or_std))
                x_denormalized.append(norm)
                
        elif normalization == 'mean_std':
            if inverse:
                denorm = (i * minimum_or_std + maximum_or_mean)
                x_denormalized.append(denorm)
            else:
                norm = ((i - maximum_or_mean)/minimum_or_std)
                x_denormalized.append(norm)
        else:
            raise TypeError("Normalization name is wrong")  
            
    return x_denormalized

def create_padding_mask(size_of_tensor: list, length_of_padding: tc.tensor, device: tc.device):
        mask = tc.ones(size_of_tensor, device = device)
        columns = tc.arange(size_of_tensor[1], device = device)
        where_to_fill = columns>=(size_of_tensor[1]-length_of_padding)
        where_to_fill = where_to_fill[(...,) + (None,) * (len(size_of_tensor)-2)].expand(size_of_tensor).to(device)
        mask = mask.masked_fill(where_to_fill, 0.0)
        return mask


def auto_encoding_MSE(input: list, target: list, length_of_padding: tc.tensor = None, is_denormalized_validation = False):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    epsilon = 1e-8
    
    mse = tc.tensor([], device = device)
    mse_per_variable = []
    if is_denormalized_validation:
       input = [x.double() for x in input]
       target = [x.double() for x in target]
        
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        for count, i in enumerate(input): 
            element_loss = loss_no_reduction(i, target[count]) 
            mask = create_padding_mask( size_of_tensor=i.size(), length_of_padding=length_of_padding, device = device).bool()
            masked_loss = element_loss[mask] #flattened vector of values not masked
            if is_denormalized_validation:
                masked_target = ((target[count])**2)[mask].mean() + epsilon
                mse_per_variable.append((masked_loss/(masked_target)).sum() / mask.sum())
                mse = tc.concatenate([mse, masked_loss/masked_target]) #only send the ones not masked to mse
                
            else:
                mse_per_variable.append(masked_loss.sum() / mask.sum())
                mse = tc.concatenate([mse, masked_loss])
                
                
    else:
        for count, i in enumerate(input):
            not_reduced_mse = loss_no_reduction(i,target[count])
            
            if is_denormalized_validation:
                not_reduced_mse = (not_reduced_mse / ((target[count]**2 ).mean()+ epsilon))

            mse_per_variable.append(not_reduced_mse.mean())
            mse = tc.concatenate([mse,not_reduced_mse.flatten()])
                
    #print('mse_per_variable',mse_per_variable)
    mse_per_variable = tc.stack(mse_per_variable)
    return mse.mean(), mse_per_variable

def dynamics_MSE(input: tc.tensor, target: tc.tensor, length_of_padding: tc.tensor = None):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        mse = tc.tensor([], device = device)
        element_loss = loss_no_reduction(input, target) 
        mask = create_padding_mask( size_of_tensor=input.size(), length_of_padding=length_of_padding, device = device).bool()
        masked_loss = element_loss[mask] #flattened vector of values not masked
        return masked_loss.mean()
                
    else:
        mse = loss(input,target)
        return mse
    

    
def initialize_model_to_last_checkpoint(encoder, f, decoder, device : tc.device, path_to_checkpoint:str ):

    checkpoint = tc.load(path_to_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    f.load_state_dict(checkpoint['f'])
    decoder.load_state_dict(checkpoint['decoder'])

def initialize_parameters(model_information, encoder, decoder, f, device):
    if not model_information['is_coupled'][0] and model_information['is_coupled'][1] == 'NODE':
        checkpoint = tc.load(model_information['path_trained_AE']+'/checkpoint/check.pt', map_location=device, weights_only=False)

        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False

        params_to_optimize = [
        {'params': f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']}
    ]
        
    elif not model_information['is_coupled'][0] and model_information['is_coupled'][1] == 'AE':
        for param in f.parameters():
            param.requires_grad = False
            
        params_to_optimize = [
        {'params': encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
        {'params': decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
    ]

    elif model_information['is_coupled'][0]:
        params_to_optimize = [
        {'params': encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
        {'params': f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']},
        {'params': decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
    ]
    return params_to_optimize