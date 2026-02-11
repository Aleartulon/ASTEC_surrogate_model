import numpy as np
import torch as tc
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from src.models.AE_NODE.training.architecture import *
from torch.utils.data import Dataset, DataLoader, get_worker_info
import subprocess

def build_dataset(batch_size:int, time_window: int, data_training_path: str, data_validation_path:str, 
                  number_of_workers:int, path_to_data: str, where_to_save:str , 
                  which_normalization:str, device :tc.device, training_boundaries:list, 
                  validation_boundaries:list, all_on_gpu:bool, pin_memory: bool, 
                  indeces_training_boundaries:str, indeces_validation_boundaries:str,
                  preload_to_ram:bool):  # Add this parameter
    
    training_path = f"{data_training_path}{str(time_window)}{indeces_training_boundaries}.h5"
    validation_path = f"{data_validation_path}{str(time_window)}{indeces_validation_boundaries}.h5"
    
    # Build dataset made out of 'time_window' chunks
    subprocess.run(['python', '-m', 'src.dataset_generation.sliced_dataset.main', 
                '--t_W', str(time_window), 
                '--path_to_dataset', path_to_data, 
                '--where_to_save_data', where_to_save, 
                '--device', device, 
                '--indeces_training_boundaries', ' ,'.join(map(str, training_boundaries)),
                '--indeces_validation_boundaries', ' ,'.join(map(str, validation_boundaries))])
    tc.cuda.empty_cache()
    
    # Build dataset and dataloader with preload option
    dataset_training = ASTEC_Dataset(training_path, all_on_gpu, device, preload_to_ram=preload_to_ram)
    dataset_validation = ASTEC_Dataset(validation_path, all_on_gpu, device, preload_to_ram=preload_to_ram)
    
    length_dataset = dataset_training.size
    if batch_size > length_dataset:
        batch_size = max(1, length_dataset // 10)
    
    print('-------------------------------------------')
    print('Length dataset: ', length_dataset)
    print('Batch size: ', batch_size)
    print('-------------------------------------------')
    
    training_loader = DataLoader(dataset_training, batch_size=batch_size, num_workers=number_of_workers, 
                                 shuffle=True, drop_last=False, pin_memory=pin_memory, 
                                 prefetch_factor=4 if number_of_workers > 0 else None, 
                                 persistent_workers=True if number_of_workers > 0 else False)
    validation_loader = DataLoader(dataset_validation, batch_size=batch_size, num_workers=number_of_workers, 
                                   shuffle=True, drop_last=False, pin_memory=pin_memory, 
                                   prefetch_factor=2 if number_of_workers > 0 else None, 
                                   persistent_workers=True if number_of_workers > 0 else False)
    
    return training_loader, validation_loader
  
def save_checkpoint(encoder, f , decoder, optimizer, scheduler, epoch, loss_value, loss_coefficients_AR_latent, loss_coefficients_full_reconstruction, before_next_window_change, how_many_datasets_creations, autoregressive_step, time_of_AE, time_of_only_TF,is_AE_frozen,scaler, index_in_window,filepath):
  
    checkpoint = {
            'encoder_state_dict':encoder.state_dict(),
            'f_state_dict':f.state_dict(),
            'decoder_state_dict':decoder.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'epoch' : epoch,
            'loss_value' : loss_value,
            'loss_coefficients_AR_latent' : loss_coefficients_AR_latent,
            'loss_coefficients_full_reconstruction' : loss_coefficients_full_reconstruction,
            'before_next_window_change' : before_next_window_change,
            'how_many_datasets_creations' : how_many_datasets_creations,
            'autoregressive_step': autoregressive_step,
            'time_of_AE': time_of_AE,
            'time_of_only_TF': time_of_only_TF,
            'is_AE_frozen' : is_AE_frozen,
            'scaler_state_dict' : scaler.state_dict() if scaler is not None else None,
            'index_in_window' : index_in_window
        }
    tc.save(checkpoint, filepath)


def load_checkpoint(encoder, f, decoder, optim, scheduler, path, device ,parent_instance):
    """
    Load checkpoint with proper handling of frozen AE state
    
    Args:
        parent_instance: The AE_NODE instance (needed to rebuild optimizer if frozen)
    """
    checkpoint = tc.load(path, map_location=device, weights_only=False)
    print("\n" + "="*70)
    print("LOADING CHECKPOINT")
    print("="*70)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Load model states
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    f.load_state_dict(checkpoint['f_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Get frozen state
    is_AE_frozen = checkpoint.get('is_AE_frozen', False)
    
    if is_AE_frozen:
        print("\nCheckpoint has frozen AE - restoring frozen state...")
        
        # Freeze encoder and decoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        
        # Rebuild optimizer with only f parameters
        f_weight_decay = 0.0
        if hasattr(parent_instance, 'model_information'):
            f_weight_decay = parent_instance.model_information.get('weight_decay', {}).get('dfnn', 0.0)
        
        params_to_optimize = [
            {'params': [p for p in f.parameters() if p.requires_grad], 
             'weight_decay': f_weight_decay}
        ]
        
        # Get learning rate from checkpoint optimizer
        checkpoint_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        
        # Create new optimizer with correct structure
        new_optim = tc.optim.Adam(params_to_optimize, lr=checkpoint_lr)
        
        # Load optimizer state
        try:
            new_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load optimizer state: {e}")
            print("  Creating fresh optimizer with checkpoint learning rate")
        
        # ✅ FIX: Replace the optimizer in parent instance
        parent_instance.optim = new_optim
        
        # ✅ FIX: Recreate scheduler with the new optimizer
        gamma = parent_instance.config_training.get('gamma_lr', 0.999)
        parent_instance.scheduler = tc.optim.lr_scheduler.ExponentialLR(parent_instance.optim, gamma)
        
        # Load scheduler state if possible
        try:
            parent_instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load scheduler state: {e}")
        
        # Recreate scaler if using mixed precision
        if parent_instance.mixed_precision and tc.cuda.is_available():
            parent_instance.scaler = GradScaler('cuda')
            print("✓ New GradScaler created")
            # Load scaler state if saved
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                try:
                    parent_instance.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("✓ Scaler state loaded")
                except Exception as e:
                    print(f"⚠ Warning: Could not load scaler state: {e}")
        
        # Verify frozen state
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        f_trainable = sum(p.numel() for p in f.parameters() if p.requires_grad)
        optim_params = sum(p.numel() for group in parent_instance.optim.param_groups for p in group['params'])
        
        print(f"\nFrozen state restored:")
        print(f"  Encoder trainable: {encoder_trainable:,} (should be 0)")
        print(f"  Decoder trainable: {decoder_trainable:,} (should be 0)")
        print(f"  F trainable: {f_trainable:,}")
        print(f"  Optimizer params: {optim_params:,} (should match F)")
        print(f"  Match? {optim_params == f_trainable}")
        
    else:
        # AE not frozen - normal loading
        print("\nCheckpoint has unfrozen AE - normal loading...")
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Optimizer and scheduler loaded normally")
    
    # Load other checkpoint data
    first_epoch = checkpoint['epoch']
    loss_value = checkpoint['loss_value']
    loss_coefficients_AR_latent = checkpoint['loss_coefficients_AR_latent']
    loss_coefficients_full_reconstruction = checkpoint['loss_coefficients_full_reconstruction']
    before_next_window = checkpoint['before_next_window_change']
    datasets_count = checkpoint['how_many_datasets_creations']
    autoregressive_step = checkpoint['autoregressive_step']
    time_of_AE = checkpoint['time_of_AE']
    time_of_only_TF = checkpoint['time_of_only_TF']
    index_in_window = checkpoint['index_in_window']
    
    print(f"\nResuming from:")
    print(f"  Epoch: {first_epoch}")
    print(f"  Loss value: {loss_value:.6f}")
    print(f"  AE frozen: {is_AE_frozen}")
    print("="*70 + "\n")
    
    return first_epoch, loss_value, loss_coefficients_AR_latent, loss_coefficients_full_reconstruction, before_next_window, datasets_count, autoregressive_step, time_of_AE, time_of_only_TF, is_AE_frozen, index_in_window


class ASTEC_Dataset(Dataset):
    def __init__(self, path: str, all_on_gpu: bool, device: tc.device, preload_to_ram: bool = True):
        """
        Dataset class for ASTEC data
        
        Args:
            path: Path to HDF5 file
            all_on_gpu: Whether to load data directly to GPU (takes precedence over preload_to_ram)
            device: torch device
            preload_to_ram: If True and all_on_gpu is False, load entire dataset to RAM at initialization
        """
        self.path = path
        self.all_on_gpu = all_on_gpu
        self.device = device
        
        # Decision logic:
        # - If all_on_gpu=True: load to GPU (original behavior)
        # - If all_on_gpu=False and preload_to_ram=True: load to RAM (new behavior)
        # - If all_on_gpu=False and preload_to_ram=False: read from disk on-demand (original behavior)
        
        if not all_on_gpu and not preload_to_ram:
            with h5py.File(self.path, 'r') as f:
                self.size = f['dictionary_of_input_variables_1'].shape[0]
        
        elif all_on_gpu:
            import time
            start = time.time()
            print(f"Loading entire dataset to GPU...")
            
            with h5py.File(path, 'r') as f:
                self.dict_vars_1 = tc.from_numpy(f['dictionary_of_input_variables_1'][:]).float().to(device)
                self.dict_vars_36 = tc.from_numpy(f['dictionary_of_input_variables_36'][:]).float().to(device)
                self.dict_vars_76 = tc.from_numpy(f['dictionary_of_input_variables_76'][:]).float().to(device)
                self.lower_plenum = tc.from_numpy(f['lower_plenum'][:]).float().to(device)
                self.dict_vars_140 = tc.from_numpy(f['dictionary_of_input_variables_140'][:]).float().to(device)
                
                bc_data = tc.from_numpy(f['boundary_conditions_and_time'][:]).float().to(device)
                self.boundary_conditions = bc_data[:, :, :-2]
                self.dt = bc_data[:, :, -2]
                
                self.length_of_padding = tc.from_numpy(f['length_of_padding'][:]).float().to(device)
                
                self.size = self.dict_vars_1.shape[0]
            
            print(f"Dataset loaded to GPU in {time.time()-start:.1f}s ({self.size} samples)")
        else: #load on RAM everything
            import time
            start = time.time()
            print(f"Pre-loading entire dataset to RAM...")
            
            with h5py.File(path, 'r') as f:
                self.dict_vars_1 = tc.from_numpy(f['dictionary_of_input_variables_1'][:]).float()
                self.dict_vars_36 = tc.from_numpy(f['dictionary_of_input_variables_36'][:]).float()
                self.dict_vars_76 = tc.from_numpy(f['dictionary_of_input_variables_76'][:]).float()
                self.lower_plenum = tc.from_numpy(f['lower_plenum'][:]).float()
                self.dict_vars_140 = tc.from_numpy(f['dictionary_of_input_variables_140'][:]).float()
                
                bc_data = tc.from_numpy(f['boundary_conditions_and_time'][:]).float()
                self.boundary_conditions = bc_data[:, :, :-2]
                self.dt = bc_data[:, :, -2]
                
                self.length_of_padding = tc.from_numpy(f['length_of_padding'][:]).float()
                
                self.size = self.dict_vars_1.shape[0]
            
            elapsed = time.time() - start
            # Calculate memory usage
            total_memory = sum([
                self.dict_vars_1.element_size() * self.dict_vars_1.nelement(),
                self.dict_vars_36.element_size() * self.dict_vars_36.nelement(),
                self.dict_vars_76.element_size() * self.dict_vars_76.nelement(),
                self.lower_plenum.element_size() * self.lower_plenum.nelement(),
                self.dict_vars_140.element_size() * self.dict_vars_140.nelement(),
                self.boundary_conditions.element_size() * self.boundary_conditions.nelement(),
                self.dt.element_size() * self.dt.nelement(),
                self.length_of_padding.element_size() * self.length_of_padding.nelement(),
            ])
            print(f"Dataset loaded to RAM in {elapsed:.1f}s ({self.size} samples, {total_memory / (1024**3):.2f} GB)")
    
    def __getitem__(self, idx):
        if hasattr(self, 'dict_vars_1'):
            return ([self.dict_vars_1[idx], 
                     self.dict_vars_36[idx], 
                     self.dict_vars_76[idx], 
                     self.lower_plenum[idx], 
                     self.dict_vars_140[idx]], 
                    self.boundary_conditions[idx], 
                    self.dt[idx], 
                    self.length_of_padding[idx])
        else:
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
            dt = tc.from_numpy(bc_data[:, -2]).float()
            
            length_of_padding = tc.from_numpy(f['length_of_padding'][idx]).float()
            
            return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, 
                    dictionary_of_input_variables_76, lower_plenum, 
                    dictionary_of_input_variables_140], boundary_conditions, dt, length_of_padding
    
    def __len__(self):
        return self.size
    
def standard_and_inverse_normalization_field(x: list, maxima_or_mean: dict, minima_or_std:dict, normalization: bool, inverse: bool):
    x_denormalized = []
    
    for _, i in enumerate(x):
        
        if i.size(-1) == 63 and len(i.size()) == 3:
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


def auto_encoding_MSE(input: list, target: list, time_series_losses_weights_AR_full_reconstruction:tc.tensor=None, length_of_padding: tc.tensor = None, is_denormalized_validation = False):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    epsilon = 1e-8
    mse = tc.tensor([], device = device)
    mse_per_variable = []
    if is_denormalized_validation:
       input = [x.double() for x in input]
       target = [x.double() for x in target]
        
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        for count, i in enumerate(input): 
            element_loss = loss_no_reduction(i, target[count])
            if time_series_losses_weights_AR_full_reconstruction is not None:
                weights = time_series_losses_weights_AR_full_reconstruction
                index_tuple = (None, slice(None)) + (None,) * (len(element_loss.size()) - 2)
                weights = weights[index_tuple]
                element_loss = element_loss * weights
            mask = create_padding_mask( size_of_tensor=i.size(), length_of_padding=length_of_padding, device = device).bool()
            if is_denormalized_validation:
                normalization = get_maxima(target[count])
                normalized_loss = element_loss**0.5 / (normalization + epsilon)
                masked_normalized_loss = normalized_loss[mask]
                mse_per_variable.append(masked_normalized_loss.sum() / mask.sum())
                mse = tc.concatenate([mse, masked_normalized_loss]) #only send the ones not masked to mse
            else:
                masked_loss = element_loss[mask]
                mse_per_variable.append(masked_loss.sum() / mask.sum())
                mse = tc.concatenate([mse, masked_loss])     
    else:
        for count, i in enumerate(input):
            not_reduced_mse = loss_no_reduction(i,target[count])
            if time_series_losses_weights_AR_full_reconstruction is not None:
                weights = time_series_losses_weights_AR_full_reconstruction
                index_tuple = (None, slice(None)) + (None,) * (len(not_reduced_mse.size()) - 2)
                weights = weights[index_tuple]
                not_reduced_mse = not_reduced_mse * weights
            if is_denormalized_validation:
                normalization = get_maxima(target[count])
                not_reduced_mse = not_reduced_mse**0.5 / (normalization + epsilon)
            mse_per_variable.append(not_reduced_mse.mean())
            mse = tc.concatenate([mse,not_reduced_mse.flatten()])
    mse_per_variable = tc.stack(mse_per_variable)
    return mse.mean(), mse_per_variable

def get_maxima(target: tc.tensor):
    size = target.size()
    where_to_contract = tuple(np.arange(3,len(target.size()),1))
    maxima = tc.amax(target, (1,) + where_to_contract) 
    if len(size) > 3:
        maxima = maxima[:, None,:,None,None]
    else:
        maxima = maxima[:, None,:]
    return maxima
    
def dynamics_MSE(input: tc.tensor, target: tc.tensor, time_series_losses_weights:tc.tensor, AR:bool, length_of_padding: tc.tensor = None):
    device = input[0].device
    loss_no_reduction = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    if (length_of_padding is not None) and tc.any(length_of_padding != 0.0):
        mse = tc.tensor([], device = device)
        element_loss = loss_no_reduction(input, target) 
        mask = create_padding_mask( size_of_tensor=input.size(), length_of_padding=length_of_padding, device = device).bool()
        masked_loss = element_loss[mask] #flattened vector of values not masked
        if AR:
            if time_series_losses_weights is not None:
                masked_loss = masked_loss.mean(-1) * time_series_losses_weights[None,:]
        return masked_loss.mean()
                
    else:
        if not AR:
            mse = loss(input,target)
        else:
            mse = loss_no_reduction(input,target)
            if time_series_losses_weights is not None:
                mse = mse.mean(-1) * time_series_losses_weights[None,:]
        return mse.mean()
    
def initialize_model_to_last_checkpoint(encoder, f, decoder, device : tc.device, path_to_checkpoint:str ):

    checkpoint = tc.load(path_to_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    f.load_state_dict(checkpoint['f_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Restore frozen state if needed
    if checkpoint.get('is_AE_frozen', False):
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False

def initialize_parameters(model_information, encoder, decoder, f, device):
    if not model_information['is_coupled'][0] and model_information['is_coupled'][1] == 'NODE':
        checkpoint = tc.load(model_information['path_trained_AE']+'/checkpoint/check.pt', map_location=device, weights_only=False)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

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