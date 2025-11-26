import numpy as np
import torch as tc
import pickle
import shutil
from src.models.AE_NODE.training.architecture import *
from src.models.AE_NODE.training.data_functions import *
from src.models.AE_NODE.training.training_validation_functions import Training



class AE_NODE:
    def __init__(self , config_training: dict, model_information: dict):
        
        self.config_training = config_training
        self.model_information = model_information
        
        self.device = config_training['device']
        self.epochs = config_training['epochs']
        self.PATH_logs = config_training['PATH']
        self.checkpoint = config_training['checkpoint']
        

        self.loss_coefficients = model_information['loss_coefficients'] if model_information['is_coupled'][0] else model_information['loss_coefficients_not_coupled']
        self.time_only_TF = model_information['time_only_TF']
        self.k = model_information['k']
        self.time_of_AE = model_information['time_of_AE']
        self.time_of_lr_war_up = model_information['time_of_lr_war_up']
        self.clipping = model_information['clipping']
        self.is_coupled = model_information['is_coupled']
        self.autoregressive_step = model_information['autoregressive_step']
        
        self.which_normalization = config_training['which_normalization']
        self.data_path = config_training['data_path']
        self.where_to_save_data = config_training['where_to_save_data']
        self.data_training_path = config_training['where_to_save_data'] + '/' + model_information['data_training_file']
        self.data_validation_path = config_training['where_to_save_data'] + '/' + model_information['data_validation_file']
        self.data_training_path_dynamic = config_training['where_to_save_data'] + '/' + model_information['data_training_file_dinamic']
        self.data_validation_path_dynamic = config_training['where_to_save_data'] + '/' + model_information['data_validation_file_dinamic']
        self.batch_sizes = config_training['batch_sizes']
        self.early_stopping = config_training['early_stopping']
        self.number_of_workers = config_training['number_of_workers']
        
        self.waiting_epochs_before_new_dataset_creation = config_training['waiting_epochs_before_new_dataset_creation']
        self.dynamic_dataset_generation_during_training = config_training['dynamic_dataset_generation_during_training']
        self.time_windows = config_training['time_windows']
        
        self.indeces_training_boundaries = config_training['indeces_training_boundaries']
        self.indeces_validation_boundaries = config_training['indeces_validation_boundaries']
        
        self.reinitialize_model_at_each_dataset_reshape = config_training['reinitialize_model_at_each_dataset_reshape']
        
        #save conversion name file 
        shutil.copy( self.data_path + '/rename_log.txt', self.PATH_logs + '/rename_log.txt')
        
        if len(self.batch_sizes) + len(self.waiting_epochs_before_new_dataset_creation) + len(self.time_windows) != len(self.time_windows) * 3:
            raise TypeError("Length of array of time_windows is not equal to length of array of batch_sizes or of waiting_epochs_before_new_dataset_creation")
        
        #check confi files are okay when training decoupled
        if not self.is_coupled[0] and self.is_coupled[1] == 'AE' and (self.loss_coefficients['TF'] != 0.0 or self.loss_coefficients['AR'] != 0.0): 
            raise TypeError("Inconsistent loss coefficients in loss_coefficients_not_coupled")
        elif not self.is_coupled[0] and self.is_coupled[1] == 'NODE' and (self.loss_coefficients['AE'][0] != 0.0 or self.loss_coefficients['AE'][1] != 0.0 or self.loss_coefficients['lambda_regularization']!= 0.0 or not self.dynamic_dataset_generation_during_training): 
            raise TypeError("Inconsistent loss coefficients in loss_coefficients_not_coupled, or dynamic_dataset_generation_during_training is set to false")
        
        #create datasets and dataloader for training and validation 
        if self.dynamic_dataset_generation_during_training:
            
            self.training_loader, self.validation_loader = build_dataset(self.batch_sizes[0], self.time_windows[0], self.data_training_path_dynamic, self.data_validation_path_dynamic, self.number_of_workers, self.data_path, self.where_to_save_data, self.which_normalization, self.device, self.indeces_training_boundaries, self.indeces_validation_boundaries)
        else:
            dataset_training = ASTEC_Dataset(self.data_training_path)
            self.training_loader = DataLoader(dataset_training, batch_size = self.batch_sizes[0], num_workers = self.number_of_workers, shuffle=True,drop_last=False,pin_memory=True)
        
            dataset_validation = ASTEC_Dataset(self.data_validation_path)
            self.validation_loader = DataLoader(dataset_validation, batch_size = self.batch_sizes[0], num_workers = self.number_of_workers, shuffle=True,drop_last=False,pin_memory=True)
        #get normalization information
        
        with open(f"{config_training['where_to_save_data']}/maxima_or_mean_{self.indeces_training_boundaries[0]}_{self.indeces_training_boundaries[1]}.pkl", 'rb') as f:
            self.maxima_or_mean = pickle.load(f)

        with open(f"{config_training['where_to_save_data']}/minima_or_std_{self.indeces_training_boundaries[0]}_{self.indeces_training_boundaries[1]}.pkl", 'rb') as f:
            self.minima_or_std = pickle.load(f)
            
        for key in self.maxima_or_mean:
            
            self.maxima_or_mean[key] = self.maxima_or_mean[key].to(self.device)
            self.minima_or_std[key] = self.minima_or_std[key].to(self.device)
            
        #define the ENCODER, the function f of the latent dynamics and the Decoder 
        self.encoder = Encoder(config_training, model_information)
        self.decoder = Decoder(config_training, model_information)
        self.f = F_Latent(config_training, model_information)

        #depending on whether the system is coupled, define f and load encoder and decoder
        if not model_information['is_coupled'][0] and model_information['is_coupled'][1] == 'NODE':
            checkpoint = tc.load(model_information['path_trained_AE']+'/checkpoint/check.pt', map_location=self.device, weights_only=False)

            self.encoder.load_state_dict(checkpoint['enco'])
            self.decoder.load_state_dict(checkpoint['dec'])

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

            params_to_optimize = [
            {'params': self.f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']}
        ]
            
        elif not model_information['is_coupled'][0] and model_information['is_coupled'][1] == 'AE':
            for param in self.f.parameters():
                param.requires_grad = False
                
            params_to_optimize = [
            {'params': self.encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
            {'params': self.decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
        ]

        elif model_information['is_coupled'][0]:
            params_to_optimize = [
            {'params': self.encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
            {'params': self.f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']},
            {'params': self.decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
        ]
            
        #move the models to the device
        self.encoder.to(self.device)
        self.f.to(self.device)
        self.decoder.to(self.device)

        #define optimizer, the pre scheduler for the warmup of the model and the scheduler
        self.optim = tc.optim.Adam(params_to_optimize, lr=config_training['learning_rate'])
        lambda1 = lambda i : i / self.time_of_lr_war_up
        self.pre_scheduler = tc.optim.lr_scheduler.LambdaLR(self.optim,lambda1)
        self.scheduler = tc.optim.lr_scheduler.ExponentialLR(self.optim, config_training['gamma_lr'])
        
        for fields, _, _, _ in self.validation_loader:
            self.number_of_different_domains = len(fields)
            break
        
        self.RK = {k: tc.tensor([[self.safe_eval(val) for val in row] for row in v]) for k, v in model_information['RK'].items()}
        
        #training starts
        
    def start_training(self):
        training_process = Training(self)
        training_process.training()
        
    def safe_eval(self, val):
        if isinstance(val, str):
            return eval(val)
        return val
        