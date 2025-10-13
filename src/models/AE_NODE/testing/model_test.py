import torch as tc
import os
import sys
from ..training.architecture import Encoder, Decoder, F_Latent, Fully_Connected_Encoder, Convolutional_Encoder


class Model_Test:
    def __init__(self , information: dict):
        
        self.path_to_test_data = information['path_to_test_data']
        self.path_to_model = information['path_to_model']
        directory_images = self.path_to_model+'/Images/'
        os.makedirs(directory_images, exist_ok=True)
        
    def test(self):
        return 0
