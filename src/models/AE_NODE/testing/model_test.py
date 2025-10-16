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
from src.models.AE_NODE.testing.support_functions import *
import matplotlib.pyplot as plt

class Model_Test:
    def __init__(self , information: dict):
        
        self.path_to_test_data = information['path_to_test_data']
        self.name_test_file = information['name_test_file']
        self.path_to_model = information['path_to_model']
        
        self.directory_images = self.path_to_model+'/Images/'
        os.makedirs(self.directory_images, exist_ok=True)
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
            error_per_trajectory_per_field, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field, final_latent_vector_per_trajectory, denormalized_fields_per_trajectory, Time = self.autoencoding()
            self.generate_pictures_autoencoding(error_per_trajectory_per_field, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field, final_latent_vector_per_trajectory, denormalized_fields_per_trajectory, Time)
            print('-----------------------------------------------------------------------')
            
    def autoencoding(self):
        
        error_per_trajectory_per_field = {'MSE_default':{}, 'MSE_normalized' : {}}
        reconstructed_fields_per_trajectory = {}
        latent_vectors_per_trajectory_per_field = {}
        final_latent_vector_per_trajectory = {}
        denormalized_fields_per_trajectory = {}
        Time = {}
        
        #access each trajectory and encode each one
        for trajectory in self.trajectories:
            fields, boundary_conditions, time = self.access_trajectory(trajectory)
            Time[trajectory] = time
            #no need to normalize because daa is already normalized in the testing

            #auto-encode
            final_latent_vector, latent_in_variables, latent_boundaries_variables, _ = self.encoder(fields, boundary_conditions)
            reconstructed_fields, reconstructed_boundary_conditions = self.decoder(final_latent_vector, latent_boundaries_variables)
            
            #give back proper shape. Not necessary if always one trajectory per time is passed but better to be general
            for count, i in enumerate(reconstructed_fields):
                size = i.size()
                reconstructed_fields[count] = tc.reshape(reconstructed_fields[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
            reconstructed_boundary_conditions = tc.reshape(reconstructed_boundary_conditions, ((boundary_conditions.size()[0],boundary_conditions.size()[1]) + reconstructed_boundary_conditions.size()[1:]))
            compute_errors_autoencoder(trajectory, error_per_trajectory_per_field, reconstructed_fields, reconstructed_boundary_conditions, fields, boundary_conditions, 'MSE_default')
            
            #de-normalize
            reconstructed_fields = standard_and_inverse_normalization_field(reconstructed_fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            reconstructed_boundary_conditions = standard_and_inverse_normalization_field([reconstructed_boundary_conditions], self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)[0]
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            boundary_conditions = standard_and_inverse_normalization_field([boundary_conditions], self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)[0]
            #compute errors
            compute_errors_autoencoder(trajectory, error_per_trajectory_per_field, reconstructed_fields, reconstructed_boundary_conditions, fields, boundary_conditions, 'MSE_normalized')
            
            #fill in dictionaries for analysis
            fill_in_dictionaries_autoencoder_step(trajectory, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field,  final_latent_vector_per_trajectory, denormalized_fields_per_trajectory,
                                                 reconstructed_fields, reconstructed_boundary_conditions, latent_in_variables,  final_latent_vector, fields, boundary_conditions)
        
            
        return error_per_trajectory_per_field, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field, final_latent_vector_per_trajectory, denormalized_fields_per_trajectory, Time
    
    
        
    
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
            time = tc.tensor(np.array(f[trajectory]['Time']), dtype=tc.float32, device = self.device)

        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], boundary_conditions, time #keep boundary conditions separated for ease

    def generate_pictures_autoencoding(self, error_per_trajectory_per_field:dict, reconstructed_fields_per_trajectory:dict, latent_vectors_per_trajectory_per_field:dict, final_latent_vector_per_trajectory:dict, denormalized_fields_per_trajectory:dict, Time:dict):
        trajectory = '1'
        
        # generate figure of dictionary_of_input_variables_1 
        plt.figure(figsize = (5,5))
        plt.plot(Time[trajectory].cpu(), reconstructed_fields_per_trajectory[trajectory][0][:,:,0].cpu()[0], label = 'AutoEncoder prediction')
        plt.plot(Time[trajectory].cpu(), denormalized_fields_per_trajectory[trajectory][0][:,:,0].cpu()[0], label = 'Ground truth')
        plt.xlabel('Time, s', fontsize = 16)
        plt.ylabel('m_cum_H2', fontsize = 16)
        plt.legend(fontsize = 16)
        plt.title('Trajectory number '+ trajectory)
        plt.savefig(self.directory_images +'/'+trajectory+'_m_cum_H2.svg', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize = (5,5))
        plt.plot(Time[trajectory].cpu(), reconstructed_fields_per_trajectory[trajectory][0][:,:,1].cpu()[0], label = 'AutoEncoder prediction')
        plt.plot(Time[trajectory].cpu(), denormalized_fields_per_trajectory[trajectory][0][:,:,1].cpu()[0], label = 'Ground truth')
        plt.xlabel('Time, s', fontsize = 16)
        plt.ylabel('m_tot_cor', fontsize = 16)
        plt.legend(fontsize = 16)
        plt.title('Trajectory number '+ trajectory)
        plt.savefig(self.directory_images +'/'+trajectory+'m_tot_cor.svg', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize = (5,5))
        plt.plot(Time[trajectory].cpu(), reconstructed_fields_per_trajectory[trajectory][0][:,:,2].cpu()[0], label = 'AutoEncoder prediction')
        plt.plot(Time[trajectory].cpu(), denormalized_fields_per_trajectory[trajectory][0][:,:,2].cpu()[0], label = 'Ground truth')
        plt.xlabel('Time, s', fontsize = 16)
        plt.ylabel('FP_A_heat', fontsize = 16)
        plt.legend(fontsize = 16)
        plt.title('Trajectory number '+ trajectory)
        plt.savefig(self.directory_images +'/'+trajectory+'FP_A_heat.svg', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize = (5,5))
        plt.plot(Time[trajectory].cpu(), reconstructed_fields_per_trajectory[trajectory][0][:,:,3].cpu()[0], label = 'AutoEncoder prediction')
        plt.plot(Time[trajectory].cpu(), denormalized_fields_per_trajectory[trajectory][0][:,:,3].cpu()[0], label = 'Ground truth')
        plt.xlabel('Time, s', fontsize = 16)
        plt.ylabel('sat_core_mesh', fontsize = 16)
        plt.legend(fontsize = 16)
        plt.title('Trajectory number '+ trajectory)
        plt.savefig(self.directory_images +'/'+trajectory+'sat_core_mesh.svg', dpi=300, bbox_inches='tight')
        
        # generate figure of core 
        print(np.shape(reconstructed_fields_per_trajectory[trajectory][1].cpu()))
        fig, axs = plt.subplots(2, 4, figsize=(8, 8))
        time_indeces = [0, 100, 1000, 20000]

        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indeces):
            all_data.append(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 0].cpu())
            all_data.append(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 0].cpu())

        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])

        # Plot with consistent color scale
        for count, i in enumerate(time_indeces):
            axs[0, count].imshow(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 0].cpu(), 
                                vmin=vmin, vmax=vmax)
            im = axs[1, count].imshow(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 0].cpu(), 
                                    vmin=vmin, vmax=vmax)

        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)

        fig.suptitle('Trajectory number ' + trajectory, fontsize=16)
        plt.savefig(self.directory_images + '/' + trajectory + 'T_comp_fuel.svg', dpi=300, bbox_inches='tight')
        
        ##################################################
        
        fig, axs = plt.subplots(2, 4, figsize=(8, 8))
        time_indeces = [0, 100, 1000, 20000]

        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indeces):
            all_data.append(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 1].cpu())
            all_data.append(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 1].cpu())

        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])

        # Plot with consistent color scale
        for count, i in enumerate(time_indeces):
            axs[0, count].imshow(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 1].cpu(), 
                                vmin=vmin, vmax=vmax)
            im = axs[1, count].imshow(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 1].cpu(), 
                                    vmin=vmin, vmax=vmax)

        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)

        fig.suptitle('Trajectory number ' + trajectory, fontsize=16)
        plt.savefig(self.directory_images + '/' + trajectory + 'T_comp_clad.svg', dpi=300, bbox_inches='tight')
        
        ##################################################
        
        fig, axs = plt.subplots(2, 4, figsize=(8, 8))
        time_indeces = [0, 100, 1000, 20000]

        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indeces):
            all_data.append(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 2].cpu())
            all_data.append(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 2].cpu())

        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])

        # Plot with consistent color scale
        for count, i in enumerate(time_indeces):
            axs[0, count].imshow(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 2].cpu(), 
                                vmin=vmin, vmax=vmax)
            im = axs[1, count].imshow(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 2].cpu(), 
                                    vmin=vmin, vmax=vmax)

        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)

        fig.suptitle('Trajectory number ' + trajectory, fontsize=16)
        plt.savefig(self.directory_images + '/' + trajectory + 'state_fuel.svg', dpi=300, bbox_inches='tight')
        
        ##################################################
        
        fig, axs = plt.subplots(2, 4, figsize=(8, 8))
        time_indeces = [0, 100, 1000, 20000]

        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indeces):
            all_data.append(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 3].cpu())
            all_data.append(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 3].cpu())

        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])

        # Plot with consistent color scale
        for count, i in enumerate(time_indeces):
            axs[0, count].imshow(reconstructed_fields_per_trajectory[trajectory][1][0, time_indeces[count], 3].cpu(), 
                                vmin=vmin, vmax=vmax)
            im = axs[1, count].imshow(denormalized_fields_per_trajectory[trajectory][1][0, time_indeces[count], 3].cpu(), 
                                    vmin=vmin, vmax=vmax)

        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)

        fig.suptitle('Trajectory number ' + trajectory, fontsize=16)
        plt.savefig(self.directory_images + '/' + trajectory + 'state_clad.svg', dpi=300, bbox_inches='tight')