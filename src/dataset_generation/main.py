import os
import sys
import torch as tc
import yaml
import shutil
from src.common_functions import load_config
from src.dataset_generation.astec_class import Astec_Dataset

def main():
    config_dataset = load_config('configs/config_dataset.yaml')
    astec_dataset = Astec_Dataset(config_dataset)
    print('---------- config_dataset ----------')
    for key, value in config_dataset.items():
        print(key, ' : ', value)
        
    testing = config_dataset['testing']
    print(f'Testing is {testing}!')
    if testing:
        #build test data
        print('--------------------------------Build testing dataset--------------------------------')
        astec_dataset.build_testing_dataset(config_dataset['indeces_testing'])
    
    else:
        
        #build training data
        print('--------------------------------Build training dataset--------------------------------')
        astec_dataset.build_training_dataset(config_dataset['indeces_training'], 'training')
        
        #build validation data
        print('--------------------------------Build validation dataset--------------------------------')
        astec_dataset.build_training_dataset(config_dataset['indeces_validation'], 'validation')

if __name__ == '__main__':
    main()
