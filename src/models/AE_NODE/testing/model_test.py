import torch as tc
import numpy as np
import os
import sys
import pickle
from ..training.architecture import Encoder, Decoder, F_Latent, Fully_Connected_Encoder, Convolutional_Encoder
from src.common_functions import load_config
import h5py
from src.dataset_generation.support_functions import normalize_fields
from src.models.AE_NODE.training.data_functions import standard_and_inverse_normalization_field

class Model_Test:
    def __init__(self , information: dict):
        
        self.path_to_test_data = information['path_to_test_data']
        self.name_test_file = information['name_test_file']
        self.path_to_model = information['path_to_model']
        
        directory_images = self.path_to_model+'/Images/'
        os.makedirs(directory_images, exist_ok=True)
        self.device = information['device']
        
        #get normalization
        
        with open(self.path_to_test_data + '/maxima_or_mean.pkl', 'rb') as file:
            self.maxima_or_mean = pickle.load(file)
        
        with open(self.path_to_test_data + '/minima_or_std.pkl', 'rb') as file:
            self.minima_or_std = pickle.load(file)
        
        for key in self.maxima_or_mean:
            self.maxima_or_mean[key] = tc.tensor(self.maxima_or_mean[key], device = self.device)
            self.minima_or_std[key] = tc.tensor(self.minima_or_std[key], device = self.device)
            
        print('Minima or std', self.minima_or_std)
        print('Maxima or mean', self.maxima_or_mean)
        
        self.models_information = load_config(self.path_to_model + 'scripts/configs/configs_models/config_AE_NODE.yaml')
        self.config_training = load_config(self.path_to_model + 'scripts/configs/config_training.yaml')
        self.which_normalization = self.config_training['which_normalization']
        
        #define models and load saved checkpoint  
        self.encoder = Encoder(self.config_training, self.models_information)
        self.f = F_Latent(self.config_training, self.models_information)
        self.decoder = Decoder(self.config_training, self.models_information)
        self.load_checkpoint_on_models()
        
        #get trajectories
        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            self.trajectories = list(f.keys())
        
    
    def test(self):
        with tc.no_grad():
            # auto-encoding verification
            print('------------------------- Purely AutoEncoding -------------------------')
            self.autoencoding()
        
    def autoencoding(self):
        
        error_per_trajectory_per_field = {}
        latent_vectors_per_trajectory_per_field = {}
        final_latent_vector_per_trajectory = {}
        
        #access each trajectory and encode each one
        for trajectory in self.trajectories:
            fields, boundary_conditions, time = self.access_trajectory(trajectory)
            #normalize before autoencoding
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = False)
            #auto-encode
            latent_in_variables, latent_boundaries_variables, regularization_latent = self.encoder(fields, boundary_conditions)
            fields, reconstructed_boundaries_variables = self.decoder(latent_in_variables, latent_boundaries_variables)
            
            #de-normalize
        
    def load_checkpoint_on_models(self):
        checkpoint = tc.load(self.path_to_model+'/checkpoint/check.pt', map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(checkpoint['enco'])
        self.f.load_state_dict(checkpoint['f'])
        self.decoder.load_state_dict(checkpoint['dec'])

        total_params_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params_f = sum(p.numel() for p in self.f.parameters() if p.requires_grad)

        memory_in_mb = (total_params_enc+total_params_dec+total_params_f * 4) / (1024 ** 2)
        print(f"Model memory: {memory_in_mb:.2f} MB")

        print(f"Total number of parameters enc: {total_params_enc}")
        print(f"Total number of parameters dec : {total_params_dec}")
        print(f"Total number of parameters input f: {total_params_f}")
        print(f"Total number of parameters: {total_params_enc+total_params_dec+total_params_f}")

        self.encoder.to(self.device)
        self.f.to(self.device)
        self.decoder.to(self.device)

        self.encoder.eval()
        self.f.eval()
        self.decoder.eval()
        
        
    def access_trajectory(self, trajectory):
        
        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            dictionary_of_input_variables_1 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_1']), dtype=tc.float32, device = self.device)
            dictionary_of_input_variables_36 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_36']), dtype=tc.float32, device = self.device)
            dictionary_of_input_variables_76 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_76']), dtype=tc.float32, device = self.device)
            lower_plenum = tc.tensor(np.array(f[trajectory]['lower_plenum']), dtype=tc.float32, device = self.device)
            dictionary_of_input_variables_140 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_140']), dtype=tc.float32, device = self.device)
            boundary_conditions = tc.tensor(np.array(f[trajectory]['boundary_conditions_and_time'][:, :,:-1]), dtype=tc.float32, device = self.device)
            time = tc.tensor(np.array(f[trajectory]['boundary_conditions_and_time'][:, :,-1]), dtype=tc.float32, device = self.device)

        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], boundary_conditions, time #keep boundary conditions separated for ease
        
    

