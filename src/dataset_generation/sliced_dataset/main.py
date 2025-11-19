import os
import sys
import torch as tc
import yaml
import shutil
import argparse
import numpy as np
from src.common_functions import load_config
from src.dataset_generation.sliced_dataset.sliced_dataset_class import Sliced_Dataset

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build sliced dataset')
    parser.add_argument('--t_W', type=int, default=None,help='Override temporal window for time subsets')
    parser.add_argument('--path_to_hdf5', type=str, default=None)
    parser.add_argument('--where_to_save_data', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_dataset = load_config('configs/config_sliced_dataset.yaml')
    
    # Override t_W if provided via command line
    if args.t_W is not None:
        config_dataset['t_W'] = args.t_W
    if args.path_to_hdf5 is not None:
        config_dataset['path_to_hdf5'] = args.path_to_hdf5
    if args.where_to_save_data is not None:
        config_dataset['where_to_save_data'] = args.where_to_save_data
    if args.device is not None:
        config_dataset['device'] = args.device
    # Initialize dataset
    sliced_dataset = Sliced_Dataset(config_dataset)
    
    print('---------- config_dataset ----------')
    for key, value in config_dataset.items():
        print(key, ' : ', value)
        
    with tc.no_grad():
        # Build training data
        print('--------------------------------Build sliced training dataset--------------------------------')
        sliced_dataset.build_sliced_training_dataset( 'training')
        tc.cuda.empty_cache()
        # Build validation data
        print('--------------------------------Build sliced validation dataset--------------------------------')
        sliced_dataset.build_sliced_training_dataset('validation')
        del sliced_dataset 
        tc.cuda.empty_cache()

if __name__ == '__main__':
    main()