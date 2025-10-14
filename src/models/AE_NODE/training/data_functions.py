import numpy as np
import torch as tc
import numpy as np
import h5py
from torch.utils.data import Dataset
    
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
        # Create a mask for values that are not -1
        mask = (i != -1)
        if i.size(-1) == 5 and len(i.size()) == 3:
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
            raise TypeError("Something is wrong with data structure")  
    
        if normalization == 'min_max':
            if inverse:
                denorm = (i * (maximum_or_mean - minimum_or_std) + minimum_or_std)
                x_denormalized.append(tc.where(mask, denorm, i))
            else:
                norm = ((i - minimum_or_std)/(maximum_or_mean - minimum_or_std))
                x_denormalized.append(tc.where(mask, norm, i))
                
        elif normalization == 'mean_std':
            if inverse:
                denorm = (i * minimum_or_std + maximum_or_mean)
                x_denormalized.append(tc.where(mask, denorm, i))
            else:
                norm = ((i - maximum_or_mean)/minimum_or_std)
                x_denormalized.append(tc.where(mask, norm, i))
            
    return x_denormalized
    

      
