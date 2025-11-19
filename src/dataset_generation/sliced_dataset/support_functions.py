import yaml
import h5py
import numpy as np
import json
import torch as tc

def dict_to_hdf5(dictionary, h5file, path=''):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_to_hdf5(value, h5file, f"{path}/{key}")
        else:
            # Handle PyTorch tensors
            if isinstance(value, tc.Tensor):
                value = value.cpu().numpy()
            # Handle other array-like objects
            elif hasattr(value, 'numpy'):
                value = value.numpy()
            
            h5file[f"{path}/{key}"] = value