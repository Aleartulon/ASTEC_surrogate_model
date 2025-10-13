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
    

      
