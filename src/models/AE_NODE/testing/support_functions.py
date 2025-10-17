import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
from src.models.AE_NODE.training.data_functions import MSE

def compute_errors_autoencoder(trajectory: str, error_per_trajectory_per_field : dict, reconstructed_fields : list, reconstructed_boundary_conditions : tc.tensor, fields : list, boundary_conditions: tc.tensor, which_error: str):
    if which_error == 'MSE_default':
        _, MSE_default = MSE(reconstructed_fields, fields, None, reconstructed_boundary_conditions, boundary_conditions, False)
        error_per_trajectory_per_field['MSE_default'][trajectory] = MSE_default
        
    elif which_error == 'MSE_normalized':
        _, MSE_normalized = MSE(reconstructed_fields, fields, None, reconstructed_boundary_conditions, boundary_conditions, True)
        error_per_trajectory_per_field['MSE_normalized'][trajectory] = MSE_normalized
    
def fill_in_dictionaries_autoencoder_step(trajectory:str, reconstructed_fields_per_trajectory:dict, latent_vectors_per_trajectory_per_field:dict,  final_latent_vector_per_trajectory:dict, denormalized_fields_per_trajectory:dict,
                                                 reconstructed_fields:list, reconstructed_boundary_conditions:tc.tensor, latent_in_variables:list, latent_boundaries_variables:tc.tensor, final_latent_vector:tc.tensor, fields: list, boundary_conditions:list):
    
    reconstructed_fields.append(reconstructed_boundary_conditions)
    fields.append(boundary_conditions)
    latent_in_variables.append(latent_boundaries_variables)
    
    reconstructed_fields_per_trajectory[trajectory] = reconstructed_fields
    latent_vectors_per_trajectory_per_field[trajectory] = latent_in_variables
    final_latent_vector_per_trajectory[trajectory] = final_latent_vector
    denormalized_fields_per_trajectory[trajectory] = fields
    
    return 0