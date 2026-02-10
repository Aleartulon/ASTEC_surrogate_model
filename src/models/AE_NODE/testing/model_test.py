import torch as tc
import numpy as np
import os
import sys
import time
import pickle
from ..training.architecture import Encoder, Decoder, F_Latent, Fully_Connected_Encoder, Convolutional_Encoder
from src.common_functions import load_config
import h5py
from src.models.AE_NODE.training.data_functions import standard_and_inverse_normalization_field
from src.models.AE_NODE.training.method_functions import Training_Losses
from src.models.AE_NODE.testing.support_functions import *
from src.dataset_generation.dataset.support_functions import build_dictionary_of_variables
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
        self.directory_images_AutoEncoding_latent_per_shape = self.directory_images + 'AutoEncoding/latent_per_shape'
        self.directory_images_AutoEncoding_errors = self.directory_images + 'AutoEncoding/errors_reconstruction_fields'
        self.directory_images_AutoEncoding_global_errors = self.directory_images + 'AutoEncoding/global_errors_reconstruction_fields'
        
        self.directory_images_TF_fields_reconstruction_scalar = self.directory_images + 'TF/fields_reconstruction_scalar'
        self.directory_images_TF_fields_reconstruction_2d = self.directory_images + 'TF/fields_reconstruction_2d'
        self.directory_images_TF_fields_reconstruction_faces = self.directory_images + 'TF/fields_reconstruction_faces'
        self.directory_images_TF_final_latent = self.directory_images + 'TF/final_latent'
        self.directory_images_TF_latent_per_shape = self.directory_images + 'TF/latent_per_shape'
        self.directory_images_TF_errors_fields = self.directory_images + 'TF/errors_reconstruction_fields'
        self.directory_images_TF_global_errors_fields = self.directory_images + 'TF/global_errors_reconstruction_fields'
        self.directory_images_TF_errors_definitive_latent = self.directory_images + 'TF/errors_reconstruction_definitive_latent'
        self.directory_images_TF_errors_latent_per_shape = self.directory_images + 'TF/errors_reconstruction_latent_per_shape'
        
        self.directory_images_AE_NODE_fields_reconstruction_scalar = self.directory_images + 'AE_NODE/fields_reconstruction_scalar'
        self.directory_images_AE_NODE_fields_reconstruction_2d = self.directory_images + 'AE_NODE/fields_reconstruction_2d'
        self.directory_images_AE_NODE_fields_reconstruction_faces = self.directory_images + 'AE_NODE/fields_reconstruction_faces'
        self.directory_images_AE_NODE_final_latent = self.directory_images + 'AE_NODE/final_latent'
        self.directory_images_AE_NODE_latent_per_shape = self.directory_images + 'AE_NODE/latent_per_shape'
        self.directory_images_AE_NODE_errors_fields = self.directory_images + 'AE_NODE/errors_reconstruction_fields'
        self.directory_images_AE_NODE_global_errors_fields = self.directory_images + 'AE_NODE/global_errors_reconstruction_fields'
        self.directory_images_AE_NODE_errors_definitive_latent = self.directory_images + 'AE_NODE/errors_reconstruction_definitive_latent'
        self.directory_images_AE_NODE_errors_latent_per_shape = self.directory_images + 'AE_NODE/errors_reconstruction_latent_per_shape'
        
        self.directory_images_Operator_Actions = self.directory_images + '/Operator_Actions'
        self.which_processor = information['which_processor']
        
        #build necessary directories to save images
        os.makedirs(self.directory_images, exist_ok=True)
        self.build_directories('AutoEncoding')
        self.build_directories('TF', AE = False)
        self.build_directories('AE_NODE', AE = False)
        os.makedirs(self.directory_images+'/Operator_Actions/', exist_ok=True)
        
        self.device = tc.device(information['device']) if tc.cuda.is_available() else tc.device("cpu")
        print('Device: ', self.device)
        self.trajectories_to_be_plotted = information['trajectories_to_be_plotted']
        for count, i in enumerate(self.trajectories_to_be_plotted):
            self.trajectories_to_be_plotted[count] = str(i)
        self.autoencoding_figures = information['autoencoding_figures']
        self.autoencoding_latent_figures = information['autoencoding_latent_figures']
        self.latent_prediction_figures = information['latent_prediction_figures']
        self.actual_latent_prediction_figures = information['actual_latent_prediction_figures']
        self.actual_fields_prediction_figures = information['actual_fields_prediction_figures']
        self.operator_actions_indeces = information['operator_actions_indeces']
        for count, i in enumerate(self.operator_actions_indeces):
            self.operator_actions_indeces[count] = str(i)
        self.TF_fields_prediction_figures = information['TF_fields_prediction_figures']
        self.TF_latent_prediction_figures = information['TF_latent_prediction_figures']
        self.compute_TF = information['compute_TF']
        
        self.models_information = load_config(self.path_to_model + 'scripts/configs/configs_models/config_AE_NODE.yaml')
        self.config_training = load_config(self.path_to_model + 'scripts/configs/config_training.yaml')
        self.which_normalization = self.config_training['which_normalization']
        self.latent_dimension = self.models_information['auto_encoding']['final_reduction_and_initial_increase']['output_dimension_encoder']
        self.config_training['device'] = self.device
        self.indeces_training_boundaries =  self.config_training['indeces_training_boundaries']
        
        self.indeces_training_boundaries = '_'
        self.generate_images_error_per_time_step = information['generate_images_error_per_time_step']
        self.generate_istograms = information["generate_istograms"]
        
        for i in self.config_training['indeces_training_boundaries']:
            self.indeces_training_boundaries += str(i) + '_'
        self.indeces_training_boundaries = self.indeces_training_boundaries[:-1]
        
        #get normalization
        with open(f"{self.path_to_test_data}/maxima_or_mean{self.indeces_training_boundaries}.pkl", 'rb') as file:
            self.maxima_or_mean = pickle.load(file)
        
        with open(f"{self.path_to_test_data}/minima_or_std{self.indeces_training_boundaries}.pkl", 'rb') as file:
            self.minima_or_std = pickle.load(file)
        
        for key in self.maxima_or_mean:
                print(f'maxima_or_mean {key}, ',self.maxima_or_mean[key])
                
        print(' ')
            
        for key in self.minima_or_std:
            print(f'minima_or_std {key}, ',self.minima_or_std[key])
                
        for key in self.maxima_or_mean:
            self.maxima_or_mean[key] = self.maxima_or_mean[key].to(self.device)
            self.minima_or_std[key] = self.minima_or_std[key].to(self.device)
            
            
        
        #define models and load saved checkpoint  
        self.encoder = Encoder(self.config_training, self.models_information)
        self.f = F_Latent(self.config_training, self.models_information)
        self.decoder = Decoder(self.config_training, self.models_information)
        self.load_checkpoint_on_models()
        
        # Check if the attribute exists and is a Parameter
        if hasattr(self.f, 'scaling_output_factor') and isinstance(self.f.scaling_output_factor[1], nn.Parameter):
            print(f"Learned scaling factor: {self.f.scaling_output_factor[1].item()}")
        
        #get trajectories
        with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
            self.trajectories = list(f.keys())
        config_processor = TrainingConfig(self.models_information, self.f, self.device )
        self.training_losses = Training_Losses(config_processor)
        
    def build_directories(self, name:str, AE:bool = False):
        os.makedirs(self.directory_images+'/' + name, exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/fields_reconstruction_scalar', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/fields_reconstruction_2d', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/fields_reconstruction_faces', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/final_latent', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/latent_per_shape', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/errors_reconstruction_fields', exist_ok=True)
        os.makedirs(self.directory_images+'/'+ name +'/global_errors_reconstruction_fields', exist_ok=True)
        
        if not AE:
            os.makedirs(self.directory_images+'/'+ name +'/errors_reconstruction_definitive_latent', exist_ok=True)
            os.makedirs(self.directory_images+'/'+ name +'/errors_reconstruction_latent_per_shape', exist_ok=True)
        
    def test(self):
        with tc.no_grad():
            # auto-encoding verification
            print('------------------------- Purely AutoEncoding -------------------------')
            reconstructed_fields_per_trajectory_AE, latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE, denormalized_fields_per_trajectory, Time = self.autoencoding()
            for trajectory_to_be_plotted in self.trajectories_to_be_plotted:
                if self.autoencoding_figures:
                    self.generate_pictures_fields(str(trajectory_to_be_plotted), reconstructed_fields_per_trajectory_AE, denormalized_fields_per_trajectory, Time, 'AE')
                    
                if self.autoencoding_latent_figures:
                    self.generate_pictures_latent_space(str(trajectory_to_be_plotted), latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE,Time, 'AE')
            
            del reconstructed_fields_per_trajectory_AE   
            #compute global errors AutoEncoding
            compute_global_errors(self.directory_images_AutoEncoding_errors, self.directory_images_AutoEncoding_global_errors, generate_istograms = self.generate_istograms, which_prediction = 'AutoEncoder')
            
            # print Operator Actions of wanted simulations
            self.print_operator_actions(definitive_latent_vector_per_trajectory_AE)
            
            # teacher forcing prediction
            if self.compute_TF:
                print('------------------------- Teacher Forcing Prediction -------------------------')  
                reconstructed_fields_per_trajectory_TF, latent_vectors_per_trajectory_per_shape_TF, definitive_latent_vector_per_trajectory_TF = self.teacher_forcing_prediction()
                for trajectory_to_be_plotted in self.trajectories_to_be_plotted:
                    if self.TF_fields_prediction_figures:
                        denormalized_fields_per_trajectory[trajectory_to_be_plotted] = [x[:,1:,:] for x in denormalized_fields_per_trajectory[trajectory_to_be_plotted]]
                        self.generate_pictures_fields(trajectory_to_be_plotted, reconstructed_fields_per_trajectory_TF, denormalized_fields_per_trajectory, Time, 'TF')
                        
                    if self.TF_latent_prediction_figures:
                        definitive_latent_vector_per_trajectory_AE[trajectory_to_be_plotted] = definitive_latent_vector_per_trajectory_AE[trajectory_to_be_plotted][1:,:]
                        latent_vectors_per_trajectory_per_shape_AE[trajectory_to_be_plotted] = [x[1:,:] for x in latent_vectors_per_trajectory_per_shape_AE[trajectory_to_be_plotted]]
                        self.generate_pictures_latent_space(trajectory_to_be_plotted, latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE, Time, 'TF', latent_vectors_per_trajectory_per_shape_TF, definitive_latent_vector_per_trajectory_TF)
                        
            del reconstructed_fields_per_trajectory_TF
            # actual prediction in latent space
            print('------------------------- Actual Prediction -------------------------')  
            reconstructed_fields_per_trajectory_AE_NODE, latent_vectors_per_trajectory_per_shape_AE_NODE, definitive_latent_vector_per_trajectory_AE_NODE = self.autoregressive_prediction()
            for trajectory_to_be_plotted in self.trajectories_to_be_plotted:
                if self.actual_fields_prediction_figures:
                    if not self.compute_TF:
                        denormalized_fields_per_trajectory[trajectory_to_be_plotted] = [x[:,1:,:] for x in denormalized_fields_per_trajectory[trajectory_to_be_plotted]]
                    self.generate_pictures_fields(trajectory_to_be_plotted, reconstructed_fields_per_trajectory_AE_NODE, denormalized_fields_per_trajectory, Time, 'AE_NODE')
                    
                if self.actual_latent_prediction_figures:
                    if not self.compute_TF:
                        definitive_latent_vector_per_trajectory_AE[trajectory_to_be_plotted] = definitive_latent_vector_per_trajectory_AE[trajectory_to_be_plotted][1:,:]
                        latent_vectors_per_trajectory_per_shape_AE[trajectory_to_be_plotted] = [x[1:,:] for x in latent_vectors_per_trajectory_per_shape_AE[trajectory_to_be_plotted]]
                    self.generate_pictures_latent_space(trajectory_to_be_plotted, latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE, Time, 'AE_NODE', latent_vectors_per_trajectory_per_shape_AE_NODE, definitive_latent_vector_per_trajectory_AE_NODE)
                    
            #compute global errors AE_NODE
            compute_global_errors(self.directory_images_AE_NODE_errors_fields, self.directory_images_AE_NODE_global_errors_fields, generate_istograms = self.generate_istograms, which_prediction = 'AE NODE')
            print('-----------------------------------------------------------------------')
            
    def print_operator_actions(self, definitive_latent_vector_per_trajectory_AE:dict):
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
                    print(f'{key:15s}: {value:12.6f} h | {desc}')
                    arr_per_operator.append(value)
                    labels.append(key)
                    info_text.append(f'{key}: {value:.2f} h | {desc}') 
            
            all_operators.append(arr_per_operator)
            all_labels.append(labels)
            all_info.append(info_text)
            
        minimum = 1e35
        maximum = -1e35
        for idx, i in enumerate(self.operator_actions_indeces):
            fig, ax = plt.subplots(figsize=(15, 4))
            with h5py.File(self.path_to_test_data + self.name_test_file, 'r') as f:
                Time = np.array(f[str(i)]['Time'])
            for dimension in range(definitive_latent_vector_per_trajectory_AE[str(i)].size(-1)):
                ax.plot(Time/ 3600.0, definitive_latent_vector_per_trajectory_AE[str(i)][:,dimension].cpu()[:])
                minimum = np.min([minimum, np.min(definitive_latent_vector_per_trajectory_AE[str(i)][:,dimension].cpu().numpy())])
                maximum = np.max([maximum, np.max(definitive_latent_vector_per_trajectory_AE[str(i)][:,dimension].cpu().numpy())])
            for count, l in enumerate(all_labels[idx]):
                y = minimum-2 + 15 if count % 2 == 0 else minimum -2 -15
                ax.annotate(l, (all_operators[idx][count], minimum-2), xytext=(0, y), textcoords='offset points', ha='center', fontsize=9)
                
            ax.scatter(all_operators[idx], np.ones_like(all_operators[idx]) * minimum, s=100, alpha=0.7)
            ax.set_xlim([-0.5, Time[-1]/3600+1])
            ax.set_ylim([minimum-2, maximum+2])
            ax.set_xlabel('Time, h', fontsize=16)
            ax.set_ylabel('Operator actions', fontsize=16)
            ax.set_title(f'Latent space evolution, trajectory {i}, and instants of operator actions')
            
            # Add text box to the right
            info_str = '\n'.join(all_info[idx])
            ax.text(0, -0.4, info_str, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.savefig(f'{self.directory_images_Operator_Actions}/{i}_operator_actions.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
    def autoencoding(self):
    
        reconstructed_fields_per_trajectory_AE = {}
        latent_vectors_per_trajectory_per_shape_AE = {}
        definitive_latent_vector_per_trajectory_AE = {}
        denormalized_fields_per_trajectory_AE = {}
        Time = {}
        
        #access each trajectory and encode each one
        for trajectory in self.trajectories:
            fields, boundary_conditions, time, _ = self.access_trajectory(trajectory)
            Time[trajectory] = time
            #no need to normalize because data is already normalized in the testing

            #auto-encode
            definitive_latent_vector, latent_in_per_shape, latent_boundaries_variables, _ = self.encoder(fields, False, boundary_conditions)
            reconstructed_fields, _ = self.decoder(definitive_latent_vector, False)
            
            #give back proper shape. Not necessary if always one trajectory per time is passed but better to be general
            for count, i in enumerate(reconstructed_fields):
                size = i.size()
                reconstructed_fields[count] = tc.reshape(reconstructed_fields[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
            
            #de-normalize
            reconstructed_fields = standard_and_inverse_normalization_field(reconstructed_fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            #compute errors
            
            #fill in dictionaries for analysis
            fill_in_dictionaries_autoencoder_step(trajectory, reconstructed_fields_per_trajectory_AE, latent_vectors_per_trajectory_per_shape_AE,  definitive_latent_vector_per_trajectory_AE, denormalized_fields_per_trajectory_AE,
                                                 reconstructed_fields, latent_in_per_shape, latent_boundaries_variables, definitive_latent_vector, fields, boundary_conditions)
            
            # compute errors
            error_per_trajectory_AE = compute_errors(trajectory, reconstructed_fields, fields, True)
            
            #print out errors in files and generate images of errors in time
            self.generate_pictures_errors_field_reconstruction(trajectory, error_per_trajectory_AE, time, self.directory_images_AutoEncoding_errors, 'AE')
        
            
        return reconstructed_fields_per_trajectory_AE, latent_vectors_per_trajectory_per_shape_AE, definitive_latent_vector_per_trajectory_AE, denormalized_fields_per_trajectory_AE, Time
    
    def teacher_forcing_prediction(self):
        reconstructed_fields_per_trajectory_TF = {}
        latent_vectors_per_trajectory_per_shape_TF = {}
        final_latent_vector_per_trajectory_TF = {}
        
        
        for trajectory in self.trajectories:
            fields, boundary_conditions, time, DT = self.access_trajectory(trajectory)
            
            #no need to normalize because data is already normalized in the testing
            
            #encode full trajectory
            definitive_latent_vectors, per_shape_latent_vectors , latent_boundaries_variables, _ = self.encoder(fields, False, boundary_conditions)
            DT = DT.unsqueeze(-1)
            
            #advance in time each time step of one dt (teacher forcing)
            advanced_latent_vectors = self.training_losses.processor(definitive_latent_vectors[:-1], DT[0][:-1], latent_boundaries_variables[:-1], 'mine')
            
            #decode back the predicted latent vectors
            reconstructed_fields, reconstructed_latent_vectors_per_field = self.decoder(advanced_latent_vectors, False)
            reconstructed_fields = [reconstructed_field.unsqueeze(0) for reconstructed_field in reconstructed_fields]
            reconstructed_fields = standard_and_inverse_normalization_field(reconstructed_fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            reconstructed_fields_per_trajectory_TF[trajectory] = reconstructed_fields
            latent_vectors_per_trajectory_per_shape_TF[trajectory] = reconstructed_latent_vectors_per_field
            final_latent_vector_per_trajectory_TF[trajectory] = advanced_latent_vectors
            
            #take only from second timestep for comparison
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            fields = [x[:,1:,:] for x in fields]
            
            definitive_latent_vectors = definitive_latent_vectors[1:,:]
            per_shape_latent_vectors = [x[1:,:] for x in per_shape_latent_vectors]
            
            # compute errors
            advanced_latent_vectors = [advanced_latent_vectors.unsqueeze(0)]
            definitive_latent_vectors = [definitive_latent_vectors.unsqueeze(0)]
            reconstructed_latent_vectors_per_field = [x.unsqueeze(0) for x in reconstructed_latent_vectors_per_field[:-1]] #-1 because it removes the None coming from the latent boundary not given (check output of Encoder from src.models.AE_NODE.training.arachitecture)
            per_shape_latent_vectors = [x.unsqueeze(0) for x in per_shape_latent_vectors[:-1]] #-1 because it removes the None coming from the latent boundary not given (check output of Encoder from src.models.AE_NODE.training.arachitecture)
            
            error_fields_per_trajectory_TF = compute_errors(trajectory, reconstructed_fields, fields, False)
            error_definitive_latent_per_trajectory_TF = compute_errors(trajectory, advanced_latent_vectors, definitive_latent_vectors, False)
            error_latent_per_variable_per_trajectory_TF = compute_errors(trajectory, reconstructed_latent_vectors_per_field, per_shape_latent_vectors, False)
            
            #print out errors in files and generate images of errors in time for field reconstruction
            self.generate_pictures_errors_field_reconstruction(trajectory, error_fields_per_trajectory_TF, time, self.directory_images_TF_errors_fields, 'TF')
            
            #print out errors in files and generate images of errors in time for field reconstruction
            self.generate_pictures_errors_latent_NODE_definitive(trajectory, error_definitive_latent_per_trajectory_TF, time)
            self.generate_pictures_errors_latent_NODE_per_shape(trajectory, error_latent_per_variable_per_trajectory_TF, time)
            
        return reconstructed_fields_per_trajectory_TF, latent_vectors_per_trajectory_per_shape_TF, final_latent_vector_per_trajectory_TF
      
    def autoregressive_prediction(self):
        reconstructed_fields_per_trajectory_AE_NODE = {}
        latent_vectors_per_trajectory_per_shape_AE_NODE = {}
        final_latent_vector_per_trajectory_AE_NODE = {}
        
        
        for trajectory in self.trajectories:
            fields, boundary_conditions, Time, DT = self.access_trajectory(trajectory)
            
            #no need to normalize because data is already normalized in the testing
            
            #encode initial condition
            t0 = time.time()
            definitive_latent_vector, per_shape_latent_vectors , latent_boundaries_variables, _ = self.encoder(fields, False, boundary_conditions)
            next_latent_vector = definitive_latent_vector[0:1]
            predicted_latents = tc.zeros((len(DT[0])-1, self.latent_dimension), device = self.device)
            
            #process in time until the end (how can I know what is the end?)
            printing = False
            t0 = time.time()
            for count, dt in enumerate(DT[0][:-1]): #last one is fake, you need one less
                if count > len(DT[0]/2) and not printing:
                    printing = True
                    t1 = time.time()
                    print(f'More than half of trajectory {trajectory} done, it took {(t1-t0)/60} minutes')
                next_latent_vector = self.training_losses.processor(next_latent_vector, dt.unsqueeze(0).unsqueeze(0), latent_boundaries_variables[count:count+1], self.which_processor)
                predicted_latents[count] = next_latent_vector
            
            #decode back the predicted latent vectors
            reconstructed_fields, reconstructed_latent_vectors_per_field = self.decoder(predicted_latents, False)
            reconstructed_fields = [reconstructed_field.unsqueeze(0) for reconstructed_field in reconstructed_fields]
            reconstructed_fields = standard_and_inverse_normalization_field(reconstructed_fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            t1 = time.time()
            print(f"Time to predict trajectory {trajectory}: {t1-t0}")
            reconstructed_fields_per_trajectory_AE_NODE[trajectory] = reconstructed_fields
            latent_vectors_per_trajectory_per_shape_AE_NODE[trajectory] = reconstructed_latent_vectors_per_field
            final_latent_vector_per_trajectory_AE_NODE[trajectory] = predicted_latents
            
            #take only from second timestep for comparison
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, inverse = True)
            fields = [x[:,1:,:] for x in fields]
            
            definitive_latent_vector = definitive_latent_vector[1:,:]
            per_shape_latent_vectors = [x[1:,:] for x in per_shape_latent_vectors]
            
            # compute errors
            predicted_latents = [predicted_latents.unsqueeze(0)]
            definitive_latent_vector = [definitive_latent_vector.unsqueeze(0)]
            reconstructed_latent_vectors_per_field = [x.unsqueeze(0) for x in reconstructed_latent_vectors_per_field[:-1]] #-1 because it removes the None coming from the latent boundary not given (check output of Encoder from src.models.AE_NODE.training.arachitecture)
            per_shape_latent_vectors = [x.unsqueeze(0) for x in per_shape_latent_vectors[:-1]] #-1 because it removes the None coming from the latent boundary not given (check output of Encoder from src.models.AE_NODE.training.arachitecture)
            
            error_fields_per_trajectory_AE_NODE = compute_errors(trajectory, reconstructed_fields, fields, False)
            error_definitive_latent_per_trajectory_AE_NODE = compute_errors(trajectory, predicted_latents, definitive_latent_vector, False)
            error_latent_per_variable_per_trajectory_AE_NODE = compute_errors(trajectory, reconstructed_latent_vectors_per_field, per_shape_latent_vectors, False)
            
            #print out errors in files and generate images of errors in time for field reconstruction
            self.generate_pictures_errors_field_reconstruction(trajectory, error_fields_per_trajectory_AE_NODE, Time, self.directory_images_AE_NODE_errors_fields, 'AE_NODE')
            
            #print out errors in files and generate images of errors in time for field reconstruction
            self.generate_pictures_errors_latent_NODE_definitive(trajectory, error_definitive_latent_per_trajectory_AE_NODE, Time)
            self.generate_pictures_errors_latent_NODE_per_shape(trajectory, error_latent_per_variable_per_trajectory_AE_NODE, Time)
            
        return reconstructed_fields_per_trajectory_AE_NODE, latent_vectors_per_trajectory_per_shape_AE_NODE, final_latent_vector_per_trajectory_AE_NODE
        
        
        
    def load_checkpoint_on_models(self):
        checkpoint = tc.load(self.path_to_model+'/checkpoint/check.pt', map_location=self.device, weights_only=False)
        # remove things added by compiler
        encoder_state_dict = {k.replace('_orig_mod.', ''): v 
                            for k, v in checkpoint['encoder_state_dict'].items()}
        f_state_dict = {k.replace('_orig_mod.', ''): v 
                        for k, v in checkpoint['f_state_dict'].items()}
        decoder_state_dict = {k.replace('_orig_mod.', ''): v 
                            for k, v in checkpoint['decoder_state_dict'].items()}
        
        self.encoder.load_state_dict(encoder_state_dict)
        self.f.load_state_dict(f_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        
        total_params_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params_f = sum(p.numel() for p in self.f.parameters() if p.requires_grad)
        memory_in_mb = (total_params_enc+total_params_dec+total_params_f) * 4 / (1024 ** 2)  # Fixed: added parentheses
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
            dictionary_of_input_variables_1 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_1']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_36 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_36']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_76 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_76']), dtype=tc.float32, device = self.device).unsqueeze(0)
            lower_plenum = tc.tensor(np.array(f[trajectory]['lower_plenum']), dtype=tc.float32, device = self.device).unsqueeze(0)
            dictionary_of_input_variables_140 = tc.tensor(np.array(f[trajectory]['dictionary_of_input_variables_140']), dtype=tc.float32, device = self.device).unsqueeze(0)
            boundary_conditions = tc.tensor(np.array(f[trajectory]['boundary_conditions_and_time'][ :,:-2]), dtype=tc.float32, device = self.device).unsqueeze(0)
            DT = tc.tensor(np.array(f[trajectory]['boundary_conditions_and_time'][ :,-2]), dtype=tc.float32, device = self.device).unsqueeze(0)
            time = tc.tensor(np.array(f[trajectory]['Time']), dtype=tc.float32, device = self.device)

        return [dictionary_of_input_variables_1, dictionary_of_input_variables_36, dictionary_of_input_variables_76, lower_plenum, dictionary_of_input_variables_140], boundary_conditions, time, DT #keep boundary conditions separated for ease
    
    def plot_scalar_values(self, trajectory, Time, reconstructed_fields, denormalized_fields, which_prediction: str,  shape_index = 0, variable_index = 0,field_name='m_cum_H2', ylabel='m_cum_H2', figsize=(5, 5), fontsize=16):
        if which_prediction == 'AE':
            index_time = 0
            label_prediction = 'AutoEncoder prediction'
            
        elif which_prediction == 'TF':
            index_time = 1
            label_prediction = 'TF prediction'
            
        elif which_prediction == 'AE_NODE':
            index_time = 1
            label_prediction = 'NODE prediction'
        else:
            raise TypeError('Wrong type of prediction')
            
        plt.figure(figsize=figsize)
        plt.plot(Time[trajectory][index_time:].cpu()[:]/ 3600.0, reconstructed_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:], 
                label=label_prediction)
        plt.plot(Time[trajectory][index_time:].cpu()[:]/ 3600.0, denormalized_fields[trajectory][shape_index][:, :, variable_index].cpu()[0][:], 
                label='Ground truth')
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}', fontsize = fontsize)
        
        if which_prediction == 'AE':
            plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_scalar}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'TF':
            plt.savefig(f'{self.directory_images_TF_fields_reconstruction_scalar}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'AE_NODE':
            plt.savefig(f'{self.directory_images_AE_NODE_fields_reconstruction_scalar}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_core_and_vessel_values(self, trajectory, Time, reconstructed_fields, denormalized_fields, which_prediction:str, field_name='state_fuel', shape_index = 0, variable_index=0, time_indices=[0, 100, 1000, 20000], figsize=(20, 8), fontsize=16, faces = False):
       
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
            
            if which_prediction == 'TF' or which_prediction == 'AE_NODE':
                axs[0, count].set_title(f't = {Time[trajectory][i+1]/3600:.2g} h', fontsize=fontsize)
                axs[1, count].set_title(f't = {Time[trajectory][i+1]/3600:.2g} h', fontsize=fontsize)
            else:
                axs[0, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)
                axs[1, count].set_title(f't = {Time[trajectory][i]/3600:.2g} h', fontsize=fontsize)
            
        axs[0, 0].set_ylabel('Prediction', fontsize=fontsize, fontweight='bold')
        axs[1, 0].set_ylabel('Ground truth', fontsize=fontsize, fontweight='bold')
        
        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axs, location='right', shrink=0.8)
        fig.suptitle(f'Trajectory number {trajectory}, {field_name}', fontsize=fontsize)
        if which_prediction == 'AE':
            if not faces:
                plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_2d}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'{self.directory_images_AutoEncoding_fields_reconstruction_faces}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'TF':
            if not faces:
                plt.savefig(f'{self.directory_images_TF_fields_reconstruction_2d}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'{self.directory_images_TF_fields_reconstruction_faces}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'AE_NODE':
            if not faces:
                plt.savefig(f'{self.directory_images_AE_NODE_fields_reconstruction_2d}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'{self.directory_images_AE_NODE_fields_reconstruction_faces}/{trajectory}_{field_name}.png', dpi=300, bbox_inches='tight')
        else:
            raise TypeError('Wrong type of prediction')
        plt.close()
        
    def plot_latent_space_per_shape(self, trajectory, Time, latent_vectors_per_trajectory_per_shape_AE, latent_vectors_per_trajectory_per_shape_AE_NODE_or_TF, which_prediction:str, shape_index = 0, ylabel='latent_plenum', figsize=(5, 5), fontsize=16):
        if which_prediction == 'AE':
            index_time = 0
        elif which_prediction == 'AE_NODE' or which_prediction == 'TF':
            index_time = 1
        else:
            raise TypeError('Wrong type of prediction')
            
        plt.figure(figsize=figsize)
        for dimension in range(latent_vectors_per_trajectory_per_shape_AE[trajectory][shape_index].size(-1)):
            color = plt.gca()._get_lines.get_next_color()
            plt.plot(Time[trajectory][index_time:].cpu()/ 3600.0, latent_vectors_per_trajectory_per_shape_AE[trajectory][shape_index][:,dimension].cpu(), label='From Encoder, dimension: ' + str(dimension+1), linestyle='--', markersize=3, color=color)
            
            if which_prediction == 'TF':
                plt.plot(Time[trajectory][index_time:].cpu()/ 3600.0, latent_vectors_per_trajectory_per_shape_AE_NODE_or_TF[trajectory][shape_index][:,dimension].cpu(), label='From TF, dimension: ' + str(dimension+1), marker='+', markersize=3, color=color)
            elif which_prediction == 'AE_NODE':
                plt.plot(Time[trajectory][index_time:].cpu()/ 3600.0, latent_vectors_per_trajectory_per_shape_AE_NODE_or_TF[trajectory][shape_index][:,dimension].cpu(), label='From NODE, dimension: ' + str(dimension+1), marker='+', markersize=3, color=color)
            
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.title(f'Trajectory number {trajectory}', fontsize = fontsize)
        
        if which_prediction == 'AE':
            plt.savefig(f'{self.directory_images_AutoEncoding_latent_per_shape}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'TF':
            plt.savefig(f'{self.directory_images_TF_latent_per_shape}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'AE_NODE':
            plt.savefig(f'{self.directory_images_AE_NODE_latent_per_shape}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_final_latent_space(self, trajectory, Time, definitive_latent_vector_per_trajectory_AE: dict, 
                            definitive_latent_vector_per_trajectory_AE_NODE_or_TF:dict, 
                            which_prediction:str, ylabel='final_latent_space', figsize=(5, 5), fontsize=16):
        if which_prediction == 'AE':
            index_time = 0
        elif which_prediction == 'AE_NODE' or which_prediction == 'TF':
            index_time = 1
        else:
            raise TypeError('Wrong type of prediction')
        
        plt.figure(figsize=figsize)
        
        # Generate unique colors using a colormap
        n_dimensions = definitive_latent_vector_per_trajectory_AE[trajectory].size(-1)
        
        # Choose appropriate colormap based on number of dimensions
        if n_dimensions <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_dimensions))
        elif n_dimensions <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_dimensions))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_dimensions))
        
        for dimension in range(n_dimensions):
            color = colors[dimension]
            plt.plot(Time[trajectory][index_time:].cpu()[:] / 3600.0, 
                    definitive_latent_vector_per_trajectory_AE[trajectory][:,dimension].cpu()[:], 
                    label='From Encoder, dimension: ' + str(dimension+1), 
                    linestyle='--', markersize=3, color=color)
            
            if which_prediction == 'TF':
                plt.plot(Time[trajectory][index_time:].cpu()[:] / 3600.0, 
                        definitive_latent_vector_per_trajectory_AE_NODE_or_TF[trajectory][:,dimension].cpu()[:], 
                        label='From NODE, dimension: ' + str(dimension+1), 
                        marker='+', markersize=3, color=color)
            elif which_prediction == 'AE_NODE':
                plt.plot(Time[trajectory][index_time:].cpu()[:] / 3600.0, 
                        definitive_latent_vector_per_trajectory_AE_NODE_or_TF[trajectory][:,dimension].cpu()[:], 
                        label='From NODE, dimension: ' + str(dimension+1), 
                        marker='+', markersize=3, color=color)
        
        plt.xlabel('Time, h', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        
        if which_prediction == 'AE':
            plt.title(f'Trajectory number {trajectory}, {definitive_latent_vector_per_trajectory_AE[trajectory].size(-1)} dimensions', fontsize=fontsize)
            plt.savefig(f'{self.directory_images_AutoEncoding_final_latent}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'TF':
            plt.title(f'Trajectory number {trajectory}, {definitive_latent_vector_per_trajectory_AE[trajectory].size(-1)} dimensions, -- from Encoder, + from TF', fontsize=fontsize)
            plt.savefig(f'{self.directory_images_TF_final_latent}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
        elif which_prediction == 'AE_NODE':
            plt.title(f'Trajectory number {trajectory}, {definitive_latent_vector_per_trajectory_AE[trajectory].size(-1)} dimensions, -- from Encoder, + from AE-NODE', fontsize=fontsize)
            plt.savefig(f'{self.directory_images_AE_NODE_final_latent}/{trajectory}_{ylabel}.png', dpi=300, bbox_inches='tight')
    
    plt.close()
            
    def generate_pictures_fields(self, trajectory_to_be_plotted:str, reconstructed_fields_per_trajectory:dict, denormalized_fields_per_trajectory:dict, Time:dict, which_prediction: str):
        
        # generate figure of global variables
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 0 ,field_name = 'm_cum_H2', ylabel =  'm cum H2', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =1 , field_name = 'm_tot_cor', ylabel = 'm tot cor', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 2 , field_name = 'FP_A_heat', ylabel = 'FP A heat', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =3 , field_name = 'sat_core_mesh', ylabel = 'sat core mesh', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 4 ,field_name = 'Q_fp_Ac', ylabel =  'm Q fp Ac', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 5 , field_name = 'Q_fp_Ag', ylabel = 'Q fp Ag', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 6 , field_name = 'Q_fp_Am', ylabel = 'Q fp Am', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 7 , field_name = 'Q_fp_As', ylabel = 'Q fp As', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 8 ,field_name = 'Q_fp_Ba', ylabel =  'Q fp Ba', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 9 , field_name = 'Q_fp_Br', ylabel = 'Q fp Br', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 10 , field_name = 'Q_fp_Cd', ylabel = 'Q fp Cd', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 11 , field_name = 'Q_fp_Ce', ylabel = 'Q fp Ce', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 12 ,field_name = 'Q_fp_Cm', ylabel =  'Q fp Cm', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 13 , field_name = 'Q_fp_Cs', ylabel = 'Q fp Cs', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 14 ,field_name = 'Q_fp_Cu', ylabel =  'Q fp Cu', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =15 , field_name = 'Q_fp_Dy', ylabel = 'Q fp Dy', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 16 , field_name = 'Q_fp_Er', ylabel = 'Q fp Er', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =17 , field_name = 'Q_fp_Eu', ylabel = 'Q fp Eu', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =18 , field_name = 'Q_fp_Ga', ylabel = 'Q fp Ga', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 19 ,field_name = 'Q_fp_Gd', ylabel =  'Q fp Gd', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =20 , field_name = 'Q_fp_Ge', ylabel = 'Q fp Ge', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 21 , field_name = 'Q_fp_Ho', ylabel = 'Q fp Ho', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 22 , field_name = 'Q_fp_I', ylabel = 'Q fp I', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 23 ,field_name = 'Q_fp_In', ylabel =  'Q fp In', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 24 , field_name = 'Q_fp_Kr', ylabel = 'Q fp Kr', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 25 ,field_name = 'Q_fp_La', ylabel =  'Q fp La', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =26, field_name = 'Q_fp_Mo', ylabel = 'Q fp Mo', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 27 , field_name = 'Q_fp_Nb', ylabel = 'Q fp Nb', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =28, field_name = 'Q_fp_Nd', ylabel = 'Q fp Nd', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 29 ,field_name = 'Q_fp_Np', ylabel =  'Q fp Np', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index =30, field_name = 'Q_fp_Pa', ylabel = 'Q fp Pa', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 31 , field_name = 'Q_fp_Pd', ylabel = 'Q fp Pd', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 32, field_name = 'Q_fp_Pm', ylabel = 'Q fp Pm', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 33 ,field_name = 'Q_fp_Pr', ylabel =  'Q fp Pr', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 34, field_name = 'Q_fp_Pu', ylabel = 'Q fp Pu', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 35 ,field_name = 'Q_fp_Ra', ylabel =  'Q fp Ra', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 36 , field_name = 'Q_fp_Rb', ylabel = 'Q fp Rb', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 37 , field_name = 'Q_fp_Re', ylabel = 'Q fp Re', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 38 , field_name = 'Q_fp_Rh', ylabel = 'Q fp Rh', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 39 ,field_name = 'Q_fp_Ru', ylabel =  'Q fp Ru', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 40 , field_name = 'Q_fp_Sb', ylabel = 'Q fp Sb', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 41 , field_name = 'Q_fp_Se', ylabel = 'Q fp Se', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 42 , field_name = 'Q_fp_Sm', ylabel = 'Q fp Sm', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 43 ,field_name = 'Q_fp_Sn', ylabel =  'Q fp Sn', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 44 , field_name = 'Q_fp_Sr', ylabel = 'Q fp Sr', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 45 ,field_name = 'Q_fp_Tb', ylabel =  'Q fp Tb', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 46 , field_name = 'Q_fp_Tc', ylabel = 'Q fp Tc', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 47 , field_name = 'Q_fp_Te', ylabel = 'Q fp Te', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 48 , field_name = 'Q_fp_Th', ylabel = 'Q fp Th', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 49 ,field_name = 'Q_fp_Tl', ylabel =  'Q fp Tl', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 50 ,field_name = 'Q_fp_Tm', ylabel =  'Q fp Tm', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 51 ,field_name = 'Q_fp_U', ylabel =  'Q fp U', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 52 , field_name = 'Q_fp_Xe', ylabel = 'Q fp Xe', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 53 , field_name = 'Q_fp_Y', ylabel = 'Q fp Y', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 54 , field_name = 'Q_fp_Yb', ylabel = 'Q fp Yb', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 55 ,field_name = 'Q_fp_Zn', ylabel =  'Q fp Zn', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 56 , field_name = 'Q_fp_Zr', ylabel = 'Q fp Zr', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 57 ,field_name = 'Q_H20_connection_primary_to_vessel', ylabel =  'Q H20 connection primary to vessel', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 58 , field_name = 'Q_steam_connection_primary_to_vessel', ylabel = 'Q steam connection primary to vessel', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 59 , field_name = 'm_H20_connection_primary_to_vessel', ylabel = 'm H20 connection primary to vessel', figsize=(5, 5), fontsize=16)
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 60 ,field_name = 'Q_H20_connection_vessel_to_primary', ylabel =  'Q H20 connection vessel to primary', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 61 , field_name = 'Q_steam_connection_vessel_to_primary', ylabel = 'Q steam connection vessel to primary', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 0, variable_index = 62 , field_name = 'm_H20_connection_vessel_to_primary', ylabel = 'm H20 connection vessel to primary', figsize=(5, 5), fontsize=16)
        
        # generate figure of core 
        time_indeces = [0, int(len(Time[trajectory_to_be_plotted])*0.4), int(len(Time[trajectory_to_be_plotted])*0.8), -2]
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='T comp fuel', shape_index = 1, variable_index=0, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='T comp clad', shape_index = 1, variable_index=1, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='state fuel', shape_index = 1, variable_index=2, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='state clad', shape_index = 1, variable_index=3, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        
        # generate figure of the vessel
        
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='P vessel', shape_index = 2, variable_index=0, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='T gas_vessel', shape_index = 2, variable_index=1, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='T liq_vessel', shape_index = 2, variable_index=2, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='x alfa vessel', shape_index = 2, variable_index=3, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='T sat vessel', shape_index = 2, variable_index=4, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='P H2 vessel', shape_index = 2, variable_index=5, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='P steam vessel', shape_index = 2, variable_index=6, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='m gas vessel', shape_index = 2, variable_index=7, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='m liq vessel mesh', shape_index = 2, variable_index=8, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='rho gas vessel', shape_index = 2, variable_index=9, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='rho liq vessel', shape_index = 2, variable_index=10, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='Q liq vap vessel', shape_index = 2, variable_index=11, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='porosity vessel', shape_index = 2, variable_index=12, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='V deb vessel', shape_index = 2, variable_index=13, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='V mag vessel', shape_index = 2, variable_index=14, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='m magma vessel', shape_index = 2, variable_index=15, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='m debris 0 vessel', shape_index = 2, variable_index=16, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='m debris 1 vessel', shape_index = 2, variable_index=17, time_indices=time_indeces, figsize=(10, 8), fontsize=16)
        
        # generate figure of lower plenum
        
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 0 ,field_name = 'P_lower_plenum', ylabel =  'P lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 1 , field_name = 'T_gas_lower_plenum', ylabel = 'T gas lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 2 , field_name = 'T_liq_lower_plenum', ylabel = 'T liq lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 3 , field_name = 'x_alfa_lower_plenum', ylabel = 'x alfa lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 4 ,field_name = 'T_sat_lower_plenum', ylabel =  'T sat lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 5 , field_name = 'P_H2_lower_plenum', ylabel = 'P H2 lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 6 , field_name = 'P_steam_lower_plenum', ylabel = 'P steam lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 7 , field_name = 'm_gas_lower_plenum', ylabel = 'm gas lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 8 ,field_name = 'm_liq_lower_plenum', ylabel =  'm liq vessel lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 9 , field_name = 'rho_gas_lower_plenum', ylabel = 'rho gas lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 10 , field_name = 'rho_liq_lower_plenum', ylabel = 'rho liq lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 11 , field_name = 'Q_liq_vap_lower_plenum', ylabel = 'Q liq vap lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 12 ,field_name = 'porosity_lower_plenum', ylabel =  'porosity lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 13 , field_name = 'V_deb_lower_plenum', ylabel = 'V deb lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 14 , field_name = 'V_mag_lower_plenum', ylabel = 'V mag lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 15 , field_name = 'm_magma_lower_plenum', ylabel = 'm magma lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 16 ,field_name = 'm_debris_0_lower_plenum', ylabel =  'm debris 0 lower plenum', figsize=(5, 5), fontsize=16)
        self.plot_scalar_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, shape_index = 3, variable_index = 17 , field_name = 'm_debris_1_lower_plenum', ylabel = 'm debris 1 lower plenum', figsize=(5, 5), fontsize=16)

        
        # generate figure of faces
        
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='Q m liq face', shape_index = 4, variable_index=0, time_indices=time_indeces, figsize=(10, 8), fontsize=16, faces = True)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='V gas face', shape_index = 4, variable_index=1, time_indices=time_indeces, figsize=(10, 8), fontsize=16, faces = True)
        self.plot_core_and_vessel_values(trajectory_to_be_plotted, Time, reconstructed_fields_per_trajectory, denormalized_fields_per_trajectory, which_prediction, field_name='V liq face', shape_index = 4, variable_index=2, time_indices=time_indeces, figsize=(10, 8), fontsize=16, faces = True)
        
    def generate_pictures_latent_space(self, trajectory_to_be_plotted, latent_vectors_per_trajectory_per_shape_AE:dict, definitive_latent_vector_per_trajectory_AE: dict, Time:dict, which_prediction: str, latent_vectors_per_trajectory_per_shape_AE_NODE:dict = None, definitive_latent_vector_per_trajectory_AE_NODE:dict = None):
            
        #save fig of latent space of scalar values
        self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE, latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 0, ylabel='latent scalar', figsize=(15, 5), fontsize=16)
        #save fig of latent space of core
        self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE, latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 1, ylabel='latent core', figsize=(15, 5), fontsize=16)
        #save fig of latent space of vessel 
        self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE, latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 2, ylabel='latent vessel', figsize=(15, 5), fontsize=16)
        #save fig of latent space of lower plenum 
        self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE, latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 3, ylabel='latent lower plenum', figsize=(15, 5), fontsize=16)
        #save fig of latent space of faces 
        self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE,  latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 4, ylabel='latent faces', figsize=(15, 5), fontsize=16)
        
        if which_prediction == 'AE': #at inference boundaries are not predicted
            #save fig of latent boundary conditions
            self.plot_latent_space_per_shape(trajectory_to_be_plotted, Time, latent_vectors_per_trajectory_per_shape_AE,  latent_vectors_per_trajectory_per_shape_AE_NODE, which_prediction, shape_index = 5, ylabel='latent boundaries', figsize=(15, 5), fontsize=16)
            
        #save fig of definitive latent vector
        self.plot_final_latent_space(trajectory_to_be_plotted, Time, definitive_latent_vector_per_trajectory_AE, definitive_latent_vector_per_trajectory_AE_NODE, which_prediction, ylabel='final latent space', figsize=(15, 5), fontsize=16)
    
    def generate_pictures_errors_field_reconstruction(self, trajectory:str, error_per_trajectory_AE:dict, time:tc.tensor, saving_directory: str, which_prediction: str): 
        dictionary_of_variables = build_dictionary_of_variables()
        scalar_variables = [ key+'_scalar' for key in dictionary_of_variables['dictionary_of_input_variables_1']]
        core_variables = [key+'_core' for key in dictionary_of_variables['dictionary_of_input_variables_36']]
        vessel_variables = [key + '_vessel' for key in dictionary_of_variables['dictionary_of_input_variables_76']]
        lower_plenum_variables = [key[:-7] + '_lower_plenum' for key in dictionary_of_variables['dictionary_of_input_variables_76']]
        faces_variables = [key + '_faces' for key in dictionary_of_variables['dictionary_of_input_variables_140']]
        all_variables = (scalar_variables + core_variables + vessel_variables + lower_plenum_variables + faces_variables)
        
        dict_of_errors = {}
        list_trajectories = list(error_per_trajectory_AE.keys())
        
        for metric in error_per_trajectory_AE[list_trajectories[0]]:
            dict_of_errors[metric] = []
        
        for metric in dict_of_errors:
            if len(metric) > 4 and metric[-4:] == 'step': #WATCH OUT, convention is that the ones in time always end with step
                for arr in error_per_trajectory_AE[trajectory][metric]:
                    for error in arr:
                        for count in range(error.size(-1)):  
                            dict_of_errors[metric].append(error[:,count].cpu().numpy())
            else:
                for arr in error_per_trajectory_AE[trajectory][metric]:
                    for error in arr:
                        dict_of_errors[metric]+=tuple(error.cpu().numpy())
                    
        #first deal with global errors per trajectory independent of time-steps
        with open(saving_directory + f'/{trajectory}_global_errors.txt', 'w') as f:
            head = "Variable name\t"
            for metric in dict_of_errors:
                if not (len(metric) > 4 and metric[-4:] == 'step'): 
                    head += metric + '\t'
            head += "\n"
            f.write(head)
            for i in range(len(all_variables)):
                column = str(all_variables[i]) + "\t"
                for metric in dict_of_errors:
                    if not (len(metric) > 4 and metric[-4:] == 'step'): 
                        column += str(dict_of_errors[metric][i]) + '\t'
                column += "\n"
                f.write(column)
                
        if trajectory in self.trajectories_to_be_plotted:
            if which_prediction == 'AE':
                index_time = 0
            elif which_prediction == 'AE_NODE' or which_prediction == 'TF':
                index_time = 1
            else:
                raise TypeError('Wrong type of prediction')
            
            #now deal with global errors per trajectory per time-steps
            if self.generate_images_error_per_time_step:
                for count, variable_name in enumerate(all_variables):
                    plt.figure(figsize=(10,5))
                    
                    for metric in dict_of_errors:
                        if len(metric) > 4 and metric[-4:] == 'step':
                            plt.plot(time[index_time:].cpu().numpy()/3600, dict_of_errors[metric][count])
                            plt.title(variable_name, fontsize = 16)
                            plt.xlabel('Time, h', fontsize = 16)
                            plt.ylabel(metric.replace("_", " "), fontsize = 16)
                            plt.yscale('log')
                            plt.savefig(f'{saving_directory}/{trajectory}_{variable_name}_{metric}.png', dpi=300, bbox_inches='tight')
                            plt.close()
                        
    def generate_pictures_errors_latent_NODE_per_shape(self, trajectory:str, latent_error :dict, time:tc.tensor): 
        all_variables = ('scalar', 'core', 'vessel', 'lower_plenum', 'faces')
        
        MSE_normalized_by_mean = []
        for arr in latent_error[trajectory]['MSE_normalized_by_mean']:
            for error in arr:
                MSE_normalized_by_mean+=tuple(error.cpu().numpy())
                
        L2_error_norm = []
        for arr in latent_error[trajectory]['L2_error_norm']:
            for error in arr:
                L2_error_norm+=tuple(error.cpu().numpy())
            
        MSE_normalized_by_mean_per_time_step = []
        for arr in latent_error[trajectory]['MSE_normalized_by_mean_per_time_step']:
            for error in arr:
                for count in range(error.size(-1)):     
                    MSE_normalized_by_mean_per_time_step.append(error[:,count].cpu().numpy())
                    
        L2_error_norm_per_time_step = []
        for arr in latent_error[trajectory]['L2_error_norm_per_time_step']:
            for error in arr:
                for count in range(error.size(-1)): 
                    L2_error_norm_per_time_step.append(error[:,count].cpu().numpy())

        if trajectory in self.trajectories_to_be_plotted:
            #now deal with global errors per trajectory per time-steps
            for count, variable_name in enumerate(all_variables):
                plt.figure(figsize=(10,5))
                plt.plot(time[1:].cpu().numpy()/3600, MSE_normalized_by_mean_per_time_step[count])
                plt.title(variable_name, fontsize = 16)
                plt.xlabel('Time, h', fontsize = 16)
                plt.ylabel('MSE normalized by mean', fontsize = 16)
                plt.savefig(f'{self.directory_images_AE_NODE_errors_latent_per_shape}/{trajectory}_{variable_name}_MSE_normalized_by_mean_per_time_step.png', dpi=300, bbox_inches='tight')
                plt.close()
                plt.figure(figsize=(10,5))
                plt.title(variable_name, fontsize = 16)
                plt.plot(time[1:].cpu().numpy()/3600, L2_error_norm_per_time_step[count])
                plt.xlabel('Time, h', fontsize = 16)
                plt.ylabel('L2 error norm per time step', fontsize = 16)
                plt.savefig(f'{self.directory_images_AE_NODE_errors_latent_per_shape}/{trajectory}_{variable_name}_L2_error_norm_per_time_step.png', dpi=300, bbox_inches='tight')
                plt.close()
                
    def generate_pictures_errors_latent_NODE_definitive(self, trajectory:str, latent_error :dict, time:tc.tensor): 
        
        MSE_normalized_by_mean = []
        for arr in latent_error[trajectory]['MSE_normalized_by_mean']:
            for error in arr:
                MSE_normalized_by_mean+=tuple(error.cpu().numpy())
                
        L2_error_norm = []
        for arr in latent_error[trajectory]['L2_error_norm']:
            for error in arr:
                L2_error_norm+=tuple(error.cpu().numpy())
            
        MSE_normalized_by_mean_per_time_step = []
        for arr in latent_error[trajectory]['MSE_normalized_by_mean_per_time_step']:
            for error in arr:
                for count in range(error.size(-1)):     
                    MSE_normalized_by_mean_per_time_step.append(error[:,count].cpu().numpy())
                    
        L2_error_norm_per_time_step = []
        for arr in latent_error[trajectory]['L2_error_norm_per_time_step']:
            for error in arr:
                for count in range(error.size(-1)): 
                    L2_error_norm_per_time_step.append(error[:,count].cpu().numpy())
                    
        #save files with average error in latent space per dimension
        with open(self.directory_images_AE_NODE_errors_definitive_latent + f'/{trajectory}_latent_errors_averaged_across_time.txt', 'w') as f:
            f.write('Dimension\tMSE_normalized_by_mean\tL2_error_norm\n')
            for i in range(len(MSE_normalized_by_mean)):
                f.write(f'{i}\t{MSE_normalized_by_mean[i]}\t{L2_error_norm[i]}\n') 
        
        if trajectory in self.trajectories_to_be_plotted:
            #now deal with global errors per trajectory per time-steps
            plt.figure(figsize=(10,5))
            for i in range(len(MSE_normalized_by_mean)):
                plt.plot(time[1:].cpu().numpy()/3600, np.array(MSE_normalized_by_mean_per_time_step)[i,:])
                plt.xlabel('Time, h', fontsize = 16)
                plt.ylabel('MSE normalized by mean', fontsize = 16)
            plt.savefig(f'{self.directory_images_AE_NODE_errors_definitive_latent}/{trajectory}_MSE_normalized_by_mean_per_time_step.png', dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(10,5))
            for i in range(len(MSE_normalized_by_mean)):
                plt.plot(time[1:].cpu().numpy()/3600, np.array(L2_error_norm_per_time_step)[i,:])
                plt.xlabel('Time, h', fontsize = 16)
                plt.ylabel('L2 error norm per time step', fontsize = 16)
            plt.savefig(f'{self.directory_images_AE_NODE_errors_definitive_latent}/{trajectory}_L2_error_norm_per_time_step.png', dpi=300, bbox_inches='tight')
            plt.close()