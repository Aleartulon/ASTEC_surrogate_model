import os
import sys
import torch as tc
import yaml
import shutil
import argparse
import numpy as np
from src.common_functions import load_config
from src.dataset_generation.dataset.astec_class import Astec_Dataset

def main():
    # Parse command-line arguments
    print(f"PID process: {os.getpid()}")
    parser = argparse.ArgumentParser(description='Build ASTEC dataset')
    parser.add_argument('--testing', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--path_to_hdf5', type=str, default=None)
    parser.add_argument('--where_to_save_data', type=str, default=None)
    parser.add_argument('--which_normalization', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_dataset = load_config('configs/config_dataset.yaml')
    
    # Override t_W if provided via command line
    if args.testing is not None:
        config_dataset['testing'] = args.testing
    if args.path_to_hdf5 is not None:
        config_dataset['path_to_hdf5'] = args.path_to_hdf5
    if args.where_to_save_data is not None:
        config_dataset['where_to_save_data'] = args.where_to_save_data
    if args.which_normalization is not None:
        config_dataset['which_normalization'] = args.which_normalization
    if args.device is not None:
        config_dataset['device'] = args.device
        
    # create directory 
    os.makedirs(config_dataset['where_to_save_data'], exist_ok=True)
    
    # copy entire directory
    shutil.copytree('src/dataset_generation/dataset/', config_dataset['where_to_save_data'], dirs_exist_ok=True)
    shutil.copy('configs/config_dataset.yaml', config_dataset['where_to_save_data'])
    shutil.copy(config_dataset['path_to_hdf5'] + "/rename_log.txt", config_dataset['where_to_save_data'])
        
    # Initialize dataset
    astec_dataset = Astec_Dataset(config_dataset)
    
    print('---------- config_dataset ----------')
    for key, value in config_dataset.items():
        print(key, ' : ', value)
    
    testing = config_dataset['testing']
    print(f'Testing is {testing}!')
    with tc.no_grad():
        if testing:
            # Build test data
            print('--------------------------------Build testing dataset--------------------------------')
            astec_dataset.build_dataset(astec_dataset.indeces_testing, 'testing', astec_dataset.subsampling_indeces_testing)
            del astec_dataset 
            tc.cuda.empty_cache()
        else:
            # Build training data
            print('--------------------------------Build training dataset--------------------------------')
            astec_dataset.build_dataset(astec_dataset.indeces_training, 'training', astec_dataset.subsampling_indeces_training)
            
            # Build validation data
            print('--------------------------------Build validation dataset--------------------------------')
            astec_dataset.build_dataset(astec_dataset.indeces_validation, 'validation', astec_dataset.subsampling_indeces_validation)

if __name__ == '__main__':
    main()