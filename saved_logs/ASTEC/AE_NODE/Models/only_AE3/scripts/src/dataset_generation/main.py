import os
import sys
import torch as tc
import yaml
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.support_functions import load_config
from src.astec_class import Astec_Dataset

def main():
    information = load_config('configs/information.yaml')
    astec_dataset = Astec_Dataset(information)
    
    print('---------- INFORMATION ----------')
    for key, value in information.items():
        print(key, ' : ', value)
        
    testing = information['testing']
    
    if testing:
        #build test data
        print('--------------------------------Build testing dataset--------------------------------')
        astec_dataset.build_testing_dataset(information['indeces_testing'])
    
    else:
        
        #build training data
        print('--------------------------------Build training dataset--------------------------------')
        astec_dataset.build_training_dataset(information['indeces_training'], 'training')
        
        #build validation data
        print('--------------------------------Build validation dataset--------------------------------')
        astec_dataset.build_training_dataset(information['indeces_validation'], 'validation')

if __name__ == '__main__':
    main()
