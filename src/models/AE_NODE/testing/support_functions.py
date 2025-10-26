import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
from src.models.AE_NODE.training.data_functions import auto_encoding_MSE
from src.models.AE_NODE.training.data_functions import dynamics_MSE
from src.models.AE_NODE.training.architecture import F_Latent
from torch import nn

def compute_errors_autoencoder(trajectory: str, error_per_trajectory_per_field : dict, reconstructed_fields : list, reconstructed_boundary_conditions : tc.tensor, fields : list, boundary_conditions: tc.tensor, which_error: str):
    if which_error == 'MSE_default':
        _, MSE_default = auto_encoding_MSE(reconstructed_fields, fields, None, reconstructed_boundary_conditions, boundary_conditions, False)
        error_per_trajectory_per_field['MSE_default'][trajectory] = MSE_default
        
    elif which_error == 'MSE_normalized':
        _, MSE_normalized = auto_encoding_MSE(reconstructed_fields, fields, None, reconstructed_boundary_conditions, boundary_conditions, True)
        error_per_trajectory_per_field['MSE_normalized'][trajectory] = MSE_normalized
    
def fill_in_dictionaries_autoencoder_step(trajectory:str, reconstructed_fields_per_trajectory:dict, latent_vectors_per_trajectory_per_field:dict,  final_latent_vector_per_trajectory:dict, denormalized_fields_per_trajectory:dict,
                                                 reconstructed_fields:list, reconstructed_boundary_conditions:tc.tensor, latent_in_per_shape:list, latent_boundaries_variables:tc.tensor, definitive_latent_vector:tc.tensor, fields: list, boundary_conditions:list):
    
    reconstructed_fields.append(reconstructed_boundary_conditions)
    fields.append(boundary_conditions)
    latent_in_per_shape.append(latent_boundaries_variables)
    
    reconstructed_fields_per_trajectory[trajectory] = reconstructed_fields
    latent_vectors_per_trajectory_per_field[trajectory] = latent_in_per_shape
    final_latent_vector_per_trajectory[trajectory] = definitive_latent_vector
    denormalized_fields_per_trajectory[trajectory] = fields
    
    return 0

class TrainingConfig:
    def __init__(self, training_config:dict, f: F_Latent, device:tc.device):
        # Required for processor_First_Order
        self.k = training_config['k']  # RK order (1 for Euler method)
        self.device = device
        self.f = f  # The dynamics function f(latent, boundaries)
        self.RK = {k: tc.tensor([[safe_eval(val) for val in row] for row in v]) for k, v in training_config['RK'].items()}
        
def safe_eval( val):
    if isinstance(val, str):
        return eval(val)
    return val

def compute_errors(trajectory:str, input: list, target:list, is_AE:bool):
    dictionary_of_errors = {}
    dictionary_of_errors[str(trajectory)] = {'MSE_normalized_by_mean':[], 'L2_error_norm' : [], 'MSE_normalized_by_mean_per_time_step':[], 'L2_error_norm_per_time_step' : []}
    #compute MSE_normalized_by_mean
    for count, i in enumerate(input):
        error = MSE_normalized_by_mean(i, target[count])
        dictionary_of_errors[str(trajectory)]['MSE_normalized_by_mean'].append(error)
        
        #compute L2_error_norm
        error = L2_error_norm(i, target[count])
        dictionary_of_errors[str(trajectory)]['L2_error_norm'].append(error)
    
        #compute MSE_normalized_by_mean_per_time_step
        error = MSE_normalized_by_mean(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['MSE_normalized_by_mean_per_time_step'].append(error)
        
        #compute L2_error_norm_per_time_step
        error = L2_error_norm(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['L2_error_norm_per_time_step'].append(error)
    
    return dictionary_of_errors

def MSE_normalized_by_mean(input:list, target:list, per_time_step: bool = False):
    input = input.double()
    target = target.double()
    loss = nn.MSELoss(reduction='none')
    array_of_errors = []
    for count, i in enumerate(input):
        e = loss(i, target[count])
        norm = (target[count]**2)
        if not per_time_step:
            where_to_contract = (0,) + tuple(np.arange(2,len(i.size()),1))
            e = e.mean(dim = where_to_contract)
            norm = norm.mean(dim = where_to_contract)
        else:
            where_to_contract = tuple(np.arange(2,len(i.size()),1))
            if len(i.size())>2:
                e = e.mean(dim = where_to_contract)
                norm = norm.mean(dim = where_to_contract)
            
        array_of_errors.append(e/norm)
        
    return array_of_errors

def L2_error_norm(input:list, target:list, per_time_step: bool = False):
    array_of_errors = []
    input = input.double()
    target = target.double()
    for count, i in enumerate(input):
        if len(i.size()) <= 2:
            e = tc.abs(i - target[count]).double()
            norm = tc.abs(target[count]).double()
        else:
            e = tc.linalg.vector_norm(i - target[count], dim = tuple(np.arange(2,len(i.size()),1))).double()
            norm = tc.linalg.vector_norm(target[count], dim = tuple(np.arange(2,len(i.size()),1))).double()
        if not per_time_step:
            e = e.mean(0)
            norm = norm.mean(0)
        array_of_errors.append(e/norm)
            
    return array_of_errors
    