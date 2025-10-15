import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
from src.models.AE_NODE.training.data_functions import MSE

def compute_errors_autoencoder(trajectory: str, error_per_trajectory_per_field : dict, reconstructed_fields : list, reconstructed_boundary_conditions : tc.tensor, fields : list, boundary_conditions: tc.tensor):
    MSE_normalized = MSE(reconstructed_fields, fields, None, reconstructed_boundary_conditions, boundary_conditions, True)
    error_per_trajectory_per_field['MSE_normalized'][trajectory] = MSE_normalized