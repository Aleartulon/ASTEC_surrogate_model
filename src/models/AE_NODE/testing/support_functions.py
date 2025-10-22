import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
from src.models.AE_NODE.training.data_functions import auto_encoding_MSE
from src.models.AE_NODE.training.data_functions import dynamics_MSE
from src.models.AE_NODE.training.architecture import F_Latent

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