import os
import sys
import torch as tc
import yaml
import shutil
from src.support_functions import load_config
from src.model_test import Model_Test

def main():
    test_config = load_config('configs/test_config.yaml')
    model_test = Model_Test(test_config)
    
    print('---------- Start Testing ----------')
    for key, value in test_config.items():
        print(key, ' : ', value)
        
    model_test.test()
    

if __name__ == '__main__':
    main()
