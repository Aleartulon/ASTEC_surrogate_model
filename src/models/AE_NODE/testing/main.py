import os
import sys
import torch as tc
import yaml
import shutil
from src.common_functions import load_config
from src.models.AE_NODE.testing.model_test import Model_Test

def main():
    config_test = load_config('configs/config_test.yaml')
    model_test = Model_Test(config_test)
    
    print('---------- Start Testing ----------')
    for key, value in config_test.items():
        print(key, ' : ', value)
        
    model_test.test()
    

if __name__ == '__main__':
    main()
