import numpy as np
import torch as tc
import pickle
from src.models.ONLY_DECODER.training.architecture import *
from src.models.ONLY_DECODER.training.data_functions import *
from src.models.ONLY_DECODER.training.training_validation_functions import Training



class ONLY_DECODER:
    def __init__(self , config_training: dict, model_information: dict):
        
        self.device = config_training['device']
        self.epochs = config_training['epochs']
        self.PATH = config_training['PATH']
        self.checkpoint = config_training['checkpoint']

        self.loss_coefficients = model_information['loss_coefficients'] 
        self.clipping = model_information['clipping']
        self.data_path = config_training['data_path']
        self.data_training_path = config_training['data_path'] + '/' + model_information['data_training_file']
        self.data_validation_path = config_training['data_path'] + '/' + model_information['data_validation_file']
        self.batch_sizes = config_training['batch_sizes']
        self.early_stopping = config_training['early_stopping']
        self.number_of_workers = config_training['number_of_workers']
        self.warm_up_time = model_information['warm_up_time']

        #create datasets and dataloader for training and validation 
        dataset_training = ASTEC_Dataset(self.data_training_path)
        self.training_loader = DataLoader(dataset_training, batch_size = self.batch_sizes[0], num_workers = self.number_of_workers, shuffle=True,drop_last=True,pin_memory=True)
    
        dataset_validation = ASTEC_Dataset(self.data_validation_path)
        self.validation_loader = DataLoader(dataset_validation, batch_size = self.batch_sizes[0], num_workers = self.number_of_workers, shuffle=True,drop_last=True,pin_memory=True)
        
        #get normalization information
        
        with open(config_training['data_path']+'/maxima_or_mean.pkl', 'rb') as f:
            self.maxima_or_mean = pickle.load(f)

        with open(config_training['data_path']+'/minima_or_std.pkl', 'rb') as f:
            self.minima_or_std = pickle.load(f)
            
        for key in self.maxima_or_mean:
            
            self.maxima_or_mean[key] = tc.tensor(self.maxima_or_mean[key], device = self.device)
            self.minima_or_std[key] = tc.tensor(self.minima_or_std[key], device = self.device)
            
        self.which_normalization = config_training['which_normalization']

        #define the Decoder 
        self.decoder = Decoder(config_training, model_information)
        params_to_optimize = [{'params': self.decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}]
            
        #move the models to the device
        self.decoder.to(self.device)

        #define optimizer, the pre scheduler for the warmup of the model and the scheduler
        self.optim = tc.optim.Adam(params_to_optimize, lr=config_training['learning_rate'])
        lambda1 = lambda i : i / self.warm_up_time
        self.pre_scheduler = tc.optim.lr_scheduler.LambdaLR(self.optim,lambda1)
        self.scheduler = tc.optim.lr_scheduler.ExponentialLR(self.optim, config_training['gamma_lr'])
        
        for fields, _ in self.validation_loader:
            self.number_of_different_domains = len(fields)
            break
        
        #training starts
        
    def start_training(self):
        training_process = Training(self)
        training_process.training()
        
    def safe_eval(self, val):
        if isinstance(val, str):
            return eval(val)
        return val