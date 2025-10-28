import numpy as np
import torch as tc
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import subprocess

def build_dataset(batch_size:int, time_window: int, data_training_path: str, data_validation_path:str, number_of_workers:int, path_to_data: str):
    
    training_path = data_training_path + str(time_window) + '.h5'
    validation_path = data_validation_path +  str(time_window) + '.h5'
    
    #build dataset made out of 'time_window' chunks
    subprocess.run(['python', '-m', 'src.dataset_generation.main', '--t_W', str(time_window), '--testing', 'false', '--path_to_hdf5', path_to_data])
    
    # build dataset and dataloader
    dataset_training = ASTEC_Dataset(training_path)
    training_loader = DataLoader(dataset_training, batch_size = batch_size, num_workers = number_of_workers, shuffle=True,drop_last=True,pin_memory=True)
    
    dataset_validation = ASTEC_Dataset(validation_path)
    validation_loader = DataLoader(dataset_validation, batch_size = batch_size, num_workers = number_of_workers, shuffle=True,drop_last=True,pin_memory=True)
    
    return training_loader, validation_loader
  
def save_checkpoint(enco, f , dec, optimizer, scheduler, epoch, loss, loss_coeff_2, start_backprop,full_training_count,filepath):
  
    checkpoint = {
            'enco':enco.state_dict(),
            'f':f.state_dict(),
            'dec':dec.state_dict(),
            'optim':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch' : epoch,
            'loss' : loss,
            'loss_coeff_2': loss_coeff_2,
            'start_backprop': start_backprop,
            'full_training_count' : full_training_count
        }
    tc.save(checkpoint, filepath)


def load_checkpoint(enco, f , dec, optim, scheduler, filepath, device):

    checkpoint = tc.load(filepath, map_location=device)
    enco.load_state_dict(checkpoint['enco'])
    f.load_state_dict(checkpoint['f'])
    dec.load_state_dict(checkpoint['dec'])
    optim.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    loss_coeff_2 = checkpoint['loss_coeff_2']
    start_backprop = checkpoint['start_backprop']
    full_training_count = checkpoint['full_training_count']
        
    return enco, f , dec, optim, scheduler , epoch, loss, loss_coeff_2, start_backprop, full_training_count

class ASTEC_Dataset(Dataset):

    def __init__(self, path):
        self.path = path
        
        with h5py.File(self.path, 'r') as f:
            self.size = np.shape(f['dictionary_of_input_variables_1'])[0]
            
    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            dictionary_of_input_variables_1 = tc.tensor(f['dictionary_of_input_variables_1'][idx], dtype=tc.float32)
            dictionary_of_input_variables_36 = tc.tensor(f['dictionary_of_input_variables_36'][idx], dtype=tc.float32)
            dictionary_of_input_variables_76 = tc.tensor(f['dictionary_of_input_variables_76'][idx], dtype=tc.float32)
            lower_plenum = tc.tensor(f['lower_plenum'][idx], dtype=tc.float32)
            dictionary_of_input_variables_140 = tc.tensor(f['dictionary_of_input_variables_140'][idx], dtype=tc.float32)
            boundary_conditions = tc.tensor(f['boundary_conditions_and_time'][idx][:,:-1], dtype=tc.float32)
            time = tc.tensor(f['boundary_conditions_and_time'][idx][:,-1], dtype=tc.float32)
            length_of_padding = tc.tensor(f['length_of_padding'][idx], dtype=tc.float32)
        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], boundary_conditions, time, length_of_padding #keep boundary conditions separated for ease
    
    def __len__(self):
        
        return self.size
    
    
def standard_and_inverse_normalization_field(x: list, maxima_or_mean: dict, minima_or_std:dict, normalization: bool, inverse: bool):
    x_denormalized = []
    
    for _, i in enumerate(x):
        
        if i.size(-1) == 4 and len(i.size()) == 3:
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
        columns = tc.arange(size_of_tensor[1])
        where_to_fill = columns>=length_of_padding
        where_to_fill = where_to_fill[(...,) + (None,) * (len(size_of_tensor)-2)].expand(size_of_tensor).to(device)
        mask = mask.masked_fill(where_to_fill, 0.0)
        
        return mask


def auto_encoding_MSE(input: list, target: list, length_of_padding: tc.tensor = None, reconstructed_boundaries: tc.tensor = None, boundaries: tc.tensor = None, is_denormalized_validation = False):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    
    mse = tc.tensor([], device = device)
    mse_per_variable = []
    counting_elements = []
    
    if reconstructed_boundaries is not None:
        input = input + [reconstructed_boundaries]
        target = target + [boundaries]
        
    if is_denormalized_validation:
        for count, i in enumerate(input):
            if is_denormalized_validation:
                i = i.double()
                target[count] = target[count].double()
        
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        for count, i in enumerate(input):  
            size = i.size() 
            
            element_loss = loss_no_reduction(i, target[count]) 
            mask = create_padding_mask( size_of_tensor=i.size(), length_of_padding=length_of_padding, device = device).bool()
            masked_loss = element_loss[mask] #flattened vector of values not masked
            if is_denormalized_validation:
                masked_target = ((target[count])**2)[mask].mean()
                mse_per_variable.append((masked_loss/masked_target).sum() / mask.sum())
                mse = tc.concatenate([mse, masked_loss/masked_target]) #only send the ones not masked to mse
                
            else:
                mse_per_variable.append(masked_loss.sum() / mask.sum())
                mse = tc.concatenate([mse, masked_loss])
                
            counting_elements.append(mask.sum())
                
    else:
        for count, i in enumerate(input):
            size = i.size()
            normal_mse = loss(i,target[count])
            not_reduced_mse = loss_no_reduction(i,target[count])
            
            if is_denormalized_validation:
                normal_mse = normal_mse/(target[count]**2).mean()
                not_reduced_mse = (not_reduced_mse / (target[count]**2).mean()).flatten()
                
            mse_per_variable.append(normal_mse)
            mse = tc.concatenate([mse,not_reduced_mse.flatten()])
            
            counting_elements.append(np.prod(size))
                
    #print('mse_per_variable',mse_per_variable)
    mse_per_variable = tc.stack(mse_per_variable)
    counting_elements = tc.tensor(counting_elements, device = device)
    return mse.mean(), mse_per_variable

def dynamics_MSE(input: tc.tensor, target: tc.tensor, length_of_padding: tc.tensor = None):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    length_of_padding = tc.tensor([[197.]])
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        mse = tc.tensor([], device = device)
        element_loss = loss_no_reduction(input, target) 
        mask = create_padding_mask( size_of_tensor=input.size(), length_of_padding=length_of_padding, device = device).bool()
        masked_loss = element_loss[mask] #flattened vector of values not masked
        return masked_loss.mean()
                
    else:
        mse = loss(input,target)
        return mse
    

    
