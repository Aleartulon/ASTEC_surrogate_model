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
from src.models.AE_NODE.training.method_functions import Training_Losses
from src.models.AE_NODE.testing.support_functions import *

import matplotlib.pyplot as plt

class Model_Test:
    def __init__(self , information: dict):
        
        self.path_to_test_data = information['path_to_test_data']
        self.name_test_file = information['name_test_file']
        self.path_to_model = information['path_to_model']
        
        self.directory_images = self.path_to_model+'/Images/'
        self.directory_images_AutoEncoding_fields_reconstruction_scalar = self.directory_images + 'AutoEncoding/fields_reconstruction_scalar'
        self.directory_images_AutoEncoding_fields_reconstruction_2d = self.directory_images + 'AutoEncoding/fields_reconstruction_2d'
        self.directory_images_AutoEncoding_fields_reconstruction_faces = self.directory_images + 'AutoEncoding/fields_reconstruction_faces'
        self.directory_images_AutoEncoding_final_latent = self.directory_images + 'AutoEncoding/final_latent'
        self.directory_images_AutoEncoding_latent_per_variables = self.directory_images + 'AutoEncoding/latent_per_variables'
        self.directory_images_Operator_Actions = self.directory_images + '/Operator_Actions'
        
        os.makedirs(self.directory_images, exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding', exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding/fields_reconstruction_scalar', exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding/fields_reconstruction_2d', exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding/fields_reconstruction_faces', exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding/final_latent', exist_ok=True)
        os.makedirs(self.directory_images+'/AutoEncoding/latent_per_variables', exist_ok=True)
        os.makedirs(self.directory_images+'/Operator_Actions/', exist_ok=True)
        
        self.device = information['device']
        self.trajectory_to_be_plotted = information['trajectory_to_be_plotted']
        self.autoencoding_figures = information['autoencoding_figures']
        self.autoencoding_latent_figures = information['autoencoding_latent_figures']
        self.latent_prediction_figures = information['latent_prediction_figures']
        self.actual_latent_prediction_figures = information['actual_latent_prediction_figures']
        self.actual_fields_prediction_figures = information['actual_fields_prediction_figures']
        self.operator_actions_indeces = information['operator_actions_indeces']
        
        
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
        self.latent_dimension = self.models_information['auto_encoding']['final_reduction_and_initial_increase']['output_dimension_encoder']
        
        #define models and load saved checkpoint  
        self.encoder = Encoder(self.config_training, self.models_information)
        self.f = F_Latent(self.config_training, self.models_information)
        self.decoder = Decoder(self.config_training, self.models_information)
        self.load_checkpoint_on_models()
        
        #get trajectories
        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            self.trajectories = list(f.keys())
        config_processor_First_Order = TrainingConfig(self.models_information, self.f, self.device )
        self.training_losses = Training_Losses(config_processor_First_Order)
    
    def test(self):
        with tc.no_grad():
            #print Operator Actions of wanted simulations
            self.print_operator_actions()
                
            # auto-encoding verification
            print('------------------------- Purely AutoEncoding -------------------------')
            error_per_trajectory_per_field_AE, reconstructed_fields_per_trajectory_AE, latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE, denormalized_fields_per_trajectory_AE, Time = self.autoencoding()
            
            if self.autoencoding_figures:
                self.generate_pictures_autoencoding(error_per_trajectory_per_field_AE, reconstructed_fields_per_trajectory_AE, denormalized_fields_per_trajectory_AE, Time)
                
            if self.autoencoding_latent_figures:
                self.generate_pictures_latent_space_autoencoding(latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE,Time)
            
            # actual prediction in latent space
            print('------------------------- Actual Prediction -------------------------')  
            error_per_trajectory_per_field_AE_NODE, reconstructed_fields_per_trajectory_AE_NODE, latent_vectors_per_trajectory_per_field_AE_NODE, definitive_latent_vector_per_trajectory_AE_NODE = self.latent_prediction()
            
            if self.actual_field_prediction_figures:
                self.generate_pictures_actual_field_predictio(error_per_trajectory_per_field_AE_NODE, reconstructed_fields_per_trajectory_AE_NODE, Time)
            if self.actual_latent_prediction_figures:
                self.generate_pictures_actual_latent_predictio(error_per_trajectory_per_field_AE_NODE, latent_vectors_per_trajectory_per_field_AE_NODE, definitive_latent_vector_per_trajectory_AE_NODE, Time)
                
            print('-----------------------------------------------------------------------')
            
    def print_operator_actions(self):
        description = {
            't_fbseb': 'Change of pressurizer valve mode',
            't1_srv': 'DBA Phase: Open PORV - time opening',
            'opensrv': 'DBA Phase: Open PORV - %Opening',
            't2_srv': 'SA phase: Open completely',
            'tendssg2': 'Closing PORV after SGTR',
            'tpesp': 'RCS - Time at which the pumps are activated',
            'tpessg': 'SG-Time at which the pumps are activated',
            'tcss': 'Instant at which containment spray system is recovered',
            'p_u5': 'Instant at which filtered containment venting system is operated/sampling the containment pressure set-point',
            'tsg2tr': 'Time at which SGTR occurs in SG or (P,T) BC'
        }
        all_operators = []
        all_labels = []
        all_info = []
        
        for index in self.operator_actions_indeces:
            labels = []
            info_text = []
            
            with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
                Time = np.array(f[str(index)]['Time'])
                arr_per_operator = []
                ops = f[str(index)]['Operator_actions']
                keys = list(ops.keys())
                
                print(f'\n{"="*80}')
                print(f'Operator actions of simulation {index}:')
                print(f'{"="*80}')
                
                for key in keys:
                    dataset = ops[key]
                    value = dataset[()]
                    desc = description[key]
                    print(f'{key:15s}: {value:12.6f} | {desc}')
                    arr_per_operator.append(value)
                    labels.append(key)
                    info_text.append(f'{key}: {value:.2f} h | {desc}') 
            
            all_operators.append(arr_per_operator)
            all_labels.append(labels)
            all_info.append(info_text)
        
        for idx, i in enumerate(self.operator_actions_indeces):
            fig, ax = plt.subplots(figsize=(15, 4))
            
            ax.scatter(all_operators[idx], np.zeros_like(all_operators[idx]), s=100, alpha=0.7)
            
            for count, l in enumerate(all_labels[idx]):
                y = 15 if count % 2 == 0 else -15
                ax.annotate(l, (all_operators[idx][count], 0), xytext=(0, y), textcoords='offset points', ha='center', fontsize=9)
            
            ax.set_xlim([-0.5, Time[-1]/3600+1])
            ax.set_ylim([0.02, -0.02])
            ax.set_xlabel('Time, h', fontsize=16)
            ax.set_ylabel('Operator actions', fontsize=16)
            
            # Add text box to the right
            info_str = '\n'.join(all_info[idx])
            ax.text(0, -0.4, info_str, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.savefig(f'{self.directory_images_Operator_Actions}/{i}_operator_actions.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
    def autoencoding(self):
        
        error_per_trajectory_per_field = {'MSE_default':{}, 'MSE_normalized' : {}}
        reconstructed_fields_per_trajectory = {}
        latent_vectors_per_trajectory_per_field = {}
        definitive_latent_vector_per_trajectory = {}
        denormalized_fields_per_trajectory = {}
        Time = {}
        
        #access each trajectory and encode each one
        for trajectory in self.trajectories:
            fields, boundary_conditions, time, _ = self.access_trajectory(trajectory)
            Time[trajectory] = time
            #no need to normalize because data is already normalized in the testing

            #auto-encode
            definitive_latent_vector, latent_in_per_shape, latent_boundaries_variables, _ = self.encoder(fields, boundary_conditions)
            reconstructed_fields, reconstructed_boundary_conditions, _ = self.decoder(definitive_latent_vector, latent_boundaries_variables)
            
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
            fill_in_dictionaries_autoencoder_step(trajectory, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field,  definitive_latent_vector_per_trajectory, denormalized_fields_per_trajectory,
                                                 reconstructed_fields, reconstructed_boundary_conditions, latent_in_per_shape, latent_boundaries_variables, definitive_latent_vector, fields, boundary_conditions)
        
            
        return error_per_trajectory_per_field, reconstructed_fields_per_trajectory, latent_vectors_per_trajectory_per_field, definitive_latent_vector_per_trajectory, denormalized_fields_per_trajectory, Time
        
    def latent_prediction(self):
        reconstructed_fields_per_trajectory_AE_NODE = {}
        latent_vectors_per_trajectory_per_field_AE_NODE = {}
        final_latent_vector_per_trajectory_AE_NODE = {}
        
        for trajectory in self.trajectories:
            fields, boundary_conditions, time, DT = self.access_trajectory(trajectory)
            
            #no need to normalize because data is already normalized in the testing
            
            #encode initial condition
            definitive_latent_vector, _ , latent_boundaries_variables, _ = self.encoder(fields, boundary_conditions)
            next_latent_vector = definitive_latent_vector[0:1]
            predicted_latents = tc.zeros((len(DT[0]), self.latent_dimension), device = self.device)
            
            #process in time until the end (how can I know what is the end?)
            del definitive_latent_vector
            del fields
            for count, dt in enumerate(DT[0][:-1]): #last one is fake, you need one less
                next_latent_vector = self.training_losses.processor_First_Order(next_latent_vector, dt, latent_boundaries_variables[count:count+1])
                predicted_latents[count] = next_latent_vector
            
            #decode back the predicted latent vectors
            reconstructed_fields, reconstructed_latent_vectors_per_field = self.decoder(predicted_latents)
            reconstructed_fields = [reconstructed_field.unsqueeze(0) for reconstructed_field in reconstructed_fields]
            reconstructed_fields = standard_and_inverse_normalization_field(reconstructed_fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            reconstructed_fields_per_trajectory_AE_NODE[trajectory] = reconstructed_fields
            latent_vectors_per_trajectory_per_field_AE_NODE[trajectory] = reconstructed_latent_vectors_per_field
            
            
        return reconstructed_fields_per_trajectory_AE_NODE, latent_vectors_per_trajectory_per_field_AE_NODE, final_latent_vector_per_trajectory_AE_NODE
        
        
        
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
            DT = tc.tensor(np.array(f[trajectory]['boundary_conditions_and_time'][:, :,-1]), dtype=tc.float32, device = self.device)
            time = tc.tensor(np.array(f[trajectory]['Time']), dtype=tc.float32, device = self.device)

        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], boundary_conditions, time, DT #keep boundary conditions separated for ease
    
    def plot_scalar_values(self, trajectory, Time, reconstructed_fields, denormalized_fields,  shape_index = 0, variable_index = 0,field_name='m_cum_H2', ylabel='m_cum_H2',
                                figsize=(5, 5), fontsize=16):

        plt.figure(figsize=figsize)
        plt.plot(Time[trajectory].cpu()[:]/ 3600.0, reconstructed_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:], 
                label='AutoEncoder prediction')
        plt.plot(Time[trajectory].cpu()[:]/ 3600.0, denormalized_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:], 
                label='Ground truth')
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}', fontsize = fontsize)
        plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_scalar}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_core_and_vessel_values(self, trajectory, Time, reconstructed_fields, denormalized_fields,field_name='state_fuel', shape_index = 0, variable_index=0, time_indices=[0, 100, 1000, 20000], figsize=(20, 8), fontsize=16, faces = False):
       
        fig, axs = plt.subplots(2, len(time_indices), figsize=figsize)
        
        # Collect all image data to determine common vmin/vmax
        all_data = []
        for count, i in enumerate(time_indices):
            all_data.append(reconstructed_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu())
            all_data.append(denormalized_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu())
        
        # Find global min and max
        vmin = min([data.min() for data in all_data])
        vmax = max([data.max() for data in all_data])
        
        # Plot with consistent color scale
        for count, i in enumerate(time_indices):
            axs[0, count].imshow(reconstructed_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu(),
                                vmin=vmin, vmax=vmax)
            im = axs[1, count].imshow(denormalized_fields[trajectory][shape_index][0, time_indices[count], variable_index].cpu(),
                                    vmin=vmin, vmax=vmax)
            axs[0, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)
            axs[1, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)
        axs[0, 0].set_ylabel('Prediction', fontsize=fontsize, fontweight='bold')
        axs[1, 0].set_ylabel('Ground truth', fontsize=fontsize, fontweight='bold')
        
        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)
        fig.suptitle(f'Trajectory number {trajectory}, {field_name}', fontsize=fontsize)
        if not faces:
            plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_2d}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_faces}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_latent_space_per_variable(self, trajectory, Time, latent_space_dict:dict, shape_index = 0, ylabel='latent_plenum', figsize=(5, 5), fontsize=16):

        plt.figure(figsize=figsize)
        for dimension in range(latent_space_dict[trajectory][shape_index].size(-1)):
            plt.plot(Time[trajectory].cpu()/ 3600.0, latent_space_dict[trajectory][shape_index][:,dimension].cpu(), label='Dimension: ' + str(dimension+1), marker='+', markersize=3)
            
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}', fontsize = fontsize)
        plt.savefig(f'{self.directory_images_AutoEncoding_latent_per_variables}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_final_latent_space(self, trajectory, Time, latent_space_dict:dict, ylabel='final_latent_space', figsize=(5, 5), fontsize=16):

        plt.figure(figsize=figsize)
        for dimension in range(latent_space_dict[trajectory].size(-1)):
            plt.plot(Time[trajectory].cpu()[:]/ 3600.0, latent_space_dict[trajectory][:,dimension].cpu()[:], label='Dimension: ' + str(dimension+1), marker='+', markersize=3)
            
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}, {latent_space_dict[trajectory].size(-1)} dimensions', fontsize = fontsize)
        plt.savefig(f'{self.directory_images_AutoEncoding_final_latent}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        plt.close()
            
    def generate_pictures_autoencoding(self, error_per_trajectory_per_field:dict, reconstructed_fields_per_trajectory:dict, denormalized_fields_per_trajectory:dict, Time:dict):
        # generate figure of dictionary_of_input_variables_1 
        
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 0, variable_index = 0 ,field_name = 'm_cum_H2', ylabel =  'm_cum_H2', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 0, variable_index =1 , field_name = 'm_tot_cor', ylabel = 'm_tot_cor', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 0, variable_index = 2 , field_name = 'FP_A_heat', ylabel = 'FP_A_heat', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 0, variable_index =3 , field_name = 'sat_core_mesh', ylabel = 'sat_core_mesh', figsize=(5, 5), fontsize=16)
        
        # generate figure of core 
        
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='T_comp_fuel', shape_index = 1, variable_index=0, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='T_comp_clad', shape_index = 1, variable_index=1, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='state_fuel', shape_index = 1, variable_index=2, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='state_clad', shape_index = 1, variable_index=3, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        
        # generate figure of the vessel
        
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='P_vessel', shape_index = 2, variable_index=0, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='T_gas_vessel', shape_index = 2, variable_index=1, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='T_liq_vessel', shape_index = 2, variable_index=2, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='x_alfa_vessel', shape_index = 2, variable_index=3, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='T_sat_vessel', shape_index = 2, variable_index=4, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='P_H2_vessel', shape_index = 2, variable_index=5, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='P_steam_vessel', shape_index = 2, variable_index=6, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='m_gas_vessel', shape_index = 2, variable_index=7, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='m_liq_vessel_mesh', shape_index = 2, variable_index=8, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='rho_gas_vessel', shape_index = 2, variable_index=9, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='rho_liq_vessel', shape_index = 2, variable_index=10, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='Q_liq_vap_vessel', shape_index = 2, variable_index=11, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='porosity_vessel', shape_index = 2, variable_index=12, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='V_deb_vessel', shape_index = 2, variable_index=13, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='V_mag_vessel', shape_index = 2, variable_index=14, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='m_magma_vessel', shape_index = 2, variable_index=15, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='m_debris_0_vessel', shape_index = 2, variable_index=16, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='m_debris_1_vessel', shape_index = 2, variable_index=17, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16)
        
        # generate figure of lower plenum
        
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 3, variable_index = 0 ,field_name = 'P_lower_plenum', ylabel =  'P_lower_plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 3, variable_index =1 , field_name = 'T_gas_lower_plenum', ylabel = 'T_gas_lower_plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 3, variable_index = 2 , field_name = 'T_liq_lower_plenum', ylabel = 'T_liq_lower_plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 3, variable_index =3 , field_name = 'x_alfa_lower_plenum', ylabel = 'x_alfa_lower_plenum', figsize=(5, 5), fontsize=16)
        
        # generate figure of faces
        
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='Q_m_liq_face', shape_index = 4, variable_index=0, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16, faces = True)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='V_gas_face', shape_index = 4, variable_index=1, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16, faces = True)
        self.plot_core_and_vessel_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, field_name='V_liq_face', shape_index = 4, variable_index=2, time_indices=[0, 1000, 10000, -1], figsize=(10, 8), fontsize=16, faces = True)
        
        #generate figures for boundary conditions
        
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 0 ,field_name = 'Q_H20_connection_v_to_p', ylabel =  'Q_H20_connection_v_to_p', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 1 , field_name = 'Q_steam_connection_v_to_p', ylabel = 'Q_steam_connection_v_to_p', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 2 , field_name = 'm_H20_connection_v_to_p', ylabel = 'm_H20_connection_v_to_p', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 3 ,field_name = 'Q_H20_connection_p_to_v', ylabel =  'Q_H20_connection_p_to_v', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 4 , field_name = 'Q_steam_connection_p_to_v', ylabel = 'Q_steam_connection_p_to_v', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(self.trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, shape_index = 5, variable_index = 5 , field_name = 'm_H20_connection_p_to_v', ylabel = 'm_H20_connection_p_to_v', figsize=(5, 5), fontsize=16)
        
    def generate_pictures_latent_space_autoencoding(self, latent_vectors_per_trajectory_per_field:dict, final_latent_vector_per_trajectory:dict ,Time:dict):
        
        #save fig of latent space of scalar values
        self.plot_latent_space_per_variable(self.trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_field, shape_index = 0, ylabel='latent_scalar', figsize=(15, 5), fontsize=16)
        #save fig of latent space of core
        self.plot_latent_space_per_variable(self.trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_field, shape_index = 1, ylabel='latent_core', figsize=(15, 5), fontsize=16)
        #save fig of latent space of vessel 
        self.plot_latent_space_per_variable(self.trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_field, shape_index = 2, ylabel='latent_vessel', figsize=(15, 5), fontsize=16)
        #save fig of latent space of lower plenum 
        self.plot_latent_space_per_variable(self.trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_field, shape_index = 3, ylabel='latent_lower_plenum', figsize=(15, 5), fontsize=16)
        #save fig of latent space of faces 
        self.plot_latent_space_per_variable(self.trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_field, shape_index = 4, ylabel='latent_faces', figsize=(15, 5), fontsize=16)
            
        #save fig of definitive latent vector
        self.plot_final_latent_space(self.trajectory_to_be_plotted, Time, final_latent_vector_per_trajectory, ylabel='final_latent_space', figsize=(15, 5), fontsize=16)
        
    def generate_pictures_autoencoding(error_per_trajectory_per_field_AE_NODE:dict, reconstructed_fields_per_trajectory_AE_NODE:dict, latent_vectors_per_trajectory_per_field_AE_NODE:dict, definitive_latent_vector_per_trajectory_AE_NODE:dict, Time:dict):
        return 0
        