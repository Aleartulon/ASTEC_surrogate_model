import numpy as np
import torch as tc
import pickle
from src.models.ASTEC_surrogate.architecture import *
from src.models.ASTEC_surrogate.data_functions import *
from torch.utils.data import DataLoader


class ASTEC_surrogate:
    def __init__(self , global_information: dict, model_information: dict):
        
        self.device = global_information['device']
        self.epochs = global_information['epochs']
        self.PATH = global_information['PATH']
        self.checkpoint = global_information['checkpoint']

        self.loss_coefficients = model_information['loss_coefficients']
        self.AR_strength =  model_information['AR_strength']
        self.time_only_TF = model_information['time_only_TF']
        self.k = model_information['k']
        self.start_backprop = model_information['start_backprop']
        self.time_of_AE = model_information['time_of_AE']
        self.TBPP_dynamic = model_information['TBPP_dynamic']
        self.clipping = model_information['clipping']
        self.is_coupled = model_information['is_coupled']
        self.lambda_regularization = model_information['lambda_regularization']
        self.data_training_path = global_information['data_path'] + '/' + model_information['data_training_file']
        self.data_validation_path = global_information['data_path'] + '/' + model_information['data_validation_file']
        self.batch_size = global_information['batch_size']
        self.early_stopping = global_information['early_stopping']
        
        with open(global_information['data_path']+'/maxima_or_mean.pkl', 'rb') as f:
            self.maxima_or_mean = pickle.load(f)

        with open(global_information['data_path']+'/minima_or_std.pkl', 'rb') as f:
            self.minima_or_std = pickle.load(f)
            
        for key in self.maxima_or_mean:
            
            self.maxima_or_mean[key] = tc.tensor(self.maxima_or_mean[key], device = self.device)
            self.minima_or_std[key] = tc.tensor(self.minima_or_std[key], device = self.device)
            
        self.normalization = global_information['normalization']

        #define the ENCODER, the function f of the latent dynamics and the Decoder 
        self.encoder = Encoder(global_information, model_information)
        self.decoder = Decoder(global_information, model_information)
        self.f = F_Latent(global_information, model_information)

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
        self.optim = tc.optim.Adam(params_to_optimize, lr=global_information['learning_rate'])
        lambda1 = lambda i : i / model_information['time_of_AE']
        self.pre_scheduler = tc.optim.lr_scheduler.LambdaLR(self.optim,lambda1)
        self.scheduler = tc.optim.lr_scheduler.ExponentialLR(self.optim, global_information['gamma_lr'])
        
        #create datasets and dataloader for training and validation 
        dataset_training = ASTEC_Dataset(self.data_training_path)
        self.training_loader = DataLoader(dataset_training, batch_size = self.batch_size, num_workers=global_information['number_of_workers'], shuffle=True,drop_last=False,pin_memory=True)
        
        dataset_validation = ASTEC_Dataset(self.data_validation_path)
        self.validation_loader = DataLoader(dataset_validation, batch_size = self.batch_size, num_workers=global_information['number_of_workers'], shuffle=True,drop_last=False,pin_memory=True)
        
        for fields, _, _, _ in self.validation_loader:
            self.number_of_different_domains = len(fields)+1
            break
        self.RK = {
                '1' : tc.tensor([[0,0],[0,1]]),
                '2' : tc.tensor([[0,0,0],[1,1,0],[0, 1/2,1/2]]),
                '3' : tc.tensor([[0,0,0,0],[1/2,1/2,0,0],[1,-1,2,0],[0,1/6,2/3,1/6]]),
                '4' : tc.tensor([[0,0,0,0,0],[1/2,1/2,0,0,0],[1/2,0,1/2,0,0],[1,0,0,1,0],[0,1/6,1/3,1/3,1/6]])
                }
        
        #training starts
        if not model_information['is_coupled'][0]: 
            model_information['loss_coeff_TF_AR_together'] = model_information['loss_coeff_not_coupled']
            
    def start_training(self):
        from src.models.ASTEC_surrogate.training_validation_functions import Training
        training_process = Training(self)
        training_process.training()