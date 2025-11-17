import os
import sys
import torch as tc
import yaml
import shutil
import argparse
from src.common_functions import load_config
from src.dataset_generation.astec_class import Astec_Dataset

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build ASTEC dataset')
    parser.add_argument('--t_W', type=int, default=None,help='Override temporal window for time subsets')
    parser.add_argument('--testing', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--path_to_hdf5', type=str, default=None)
    parser.add_argument('--where_to_save_data', type=str, default=None)
    parser.add_argument('--which_normalization', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_dataset = load_config('configs/config_dataset.yaml')
    
    # Override t_W if provided via command line
    if args.t_W is not None:
        config_dataset['t_W'] = args.t_W
    if args.testing is not None:
        config_dataset['testing'] = args.testing
    if args.path_to_hdf5 is not None:
        config_dataset['path_to_hdf5'] = args.path_to_hdf5
    if args.where_to_save_data is not None:
        config_dataset['where_to_save_data'] = args.where_to_save_data
    if args.which_normalization is not None:
        config_dataset['which_normalization'] = args.which_normalization
    
    # Initialize dataset
    astec_dataset = Astec_Dataset(config_dataset)
    
    print('---------- config_dataset ----------')
    for key, value in config_dataset.items():
        print(key, ' : ', value)
    
    testing = config_dataset['testing']
    print(f'Testing is {testing}!')
    
    if testing:
        # Build test data
        print('--------------------------------Build testing dataset--------------------------------')
        astec_dataset.build_testing_dataset(config_dataset['indeces_testing'])
    else:
        # Build training data
        print('--------------------------------Build training dataset--------------------------------')
        astec_dataset.build_training_dataset(config_dataset['indeces_training'], 'training')
        
        # Build validation data
        print('--------------------------------Build validation dataset--------------------------------')
        astec_dataset.build_training_dataset(config_dataset['indeces_validation'], 'validation')

if __name__ == '__main__':
    main()