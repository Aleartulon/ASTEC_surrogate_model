import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
import pandas as pd
import os
from src.models.AE_NODE.training.data_functions import auto_encoding_MSE
from src.models.AE_NODE.training.data_functions import dynamics_MSE
from src.models.AE_NODE.training.architecture import F_Latent
from torch import nn
import matplotlib.pyplot as plt

def fill_in_dictionaries_autoencoder_step(trajectory:str, reconstructed_fields_per_trajectory:dict, latent_vectors_per_trajectory_per_field:dict,  final_latent_vector_per_trajectory:dict, denormalized_fields_per_trajectory:dict,
                                                 reconstructed_fields:list, latent_in_per_shape:list, latent_boundaries_variables:tc.tensor, definitive_latent_vector:tc.tensor, fields: list, boundary_conditions:list):
    
    latent_in_per_shape.append(latent_boundaries_variables)
    reconstructed_fields_per_trajectory[trajectory] = reconstructed_fields
    latent_vectors_per_trajectory_per_field[trajectory] = latent_in_per_shape
    final_latent_vector_per_trajectory[trajectory] = definitive_latent_vector
    denormalized_fields_per_trajectory[trajectory] = fields
    
    return 0

class TrainingConfig:
    def __init__(self, training_config:dict, f: F_Latent, device:tc.device):
        # Required for processor_First_Order
        self.k = training_config['k']  # RK order (1 for Euler method)
        self.device = device
        self.f = f  # The dynamics function f(latent, boundaries)
        self.RK = {k: tc.tensor([[safe_eval(val) for val in row] for row in v]) for k, v in training_config['RK'].items()}
        
def safe_eval( val):
    if isinstance(val, str):
        return eval(val)
    return val

def compute_errors(trajectory:str, input: list, target:list, is_AE:bool):
    dictionary_of_errors = {}
    dictionary_of_errors[str(trajectory)] = {'MSE' : [],
                                             'RMSE' : [],
                                            'MSE_normalized_by_mean':[], 
                                             'L2_error_norm' : [], 
                                             'RMSE_divided_by_max' : [],
                                             
                                             'MSE_per_time_step':[], 
                                             'RMSE_per_time_step':[], 
                                             'MSE_normalized_by_mean_per_time_step':[], 
                                             'L2_error_norm_per_time_step' : [],
                                             'RMSE_divided_by_max_per_time_step':[]
                                             }
    
    #compute MSE_normalized_by_mean
    for count, i in enumerate(input):
        #compute MSE
        error = MSE(i, target[count])
        dictionary_of_errors[str(trajectory)]['MSE'].append(error)
        
        #compute RMSE
        error = RMSE(i, target[count])
        dictionary_of_errors[str(trajectory)]['RMSE'].append(error)
        
        #compute MSE_normalized_by_mean
        error = MSE_normalized_by_mean(i, target[count])
        dictionary_of_errors[str(trajectory)]['MSE_normalized_by_mean'].append(error)
        
        #compute L2_error_norm
        error = L2_error_norm(i, target[count])
        dictionary_of_errors[str(trajectory)]['L2_error_norm'].append(error)
        
        #compute RMSE_divided_by_max
        error = RMSE_divided_by_max(i, target[count])
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_max'].append(error)
        
        ############################################################################################
    
        #compute MSE_per_time_step
        error = MSE(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['MSE_per_time_step'].append(error)
    
        #compute RMSE_per_time_step
        error = RMSE(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['RMSE_per_time_step'].append(error)
        
        #compute MSE_normalized_by_mean_per_time_step
        error = MSE_normalized_by_mean(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['MSE_normalized_by_mean_per_time_step'].append(error)
        
        #compute L2_error_norm_per_time_step
        error = L2_error_norm(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['L2_error_norm_per_time_step'].append(error)
        
        #compute RMSE_divided_by_max_per_time_step
        error = RMSE_divided_by_max(i, target[count], per_time_step = True)
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_max_per_time_step'].append(error)
    return dictionary_of_errors

def MSE(input:tc.tensor, target:tc.tensor, per_time_step: bool = False):
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)
    loss = nn.MSELoss(reduction='none')
    array_of_errors = []
    e = loss(input, target)
    where_to_contract = tuple(np.arange(2,len(input.size()),1))
    if len(input.size())>2:
        e = e.mean(dim = where_to_contract)
    if per_time_step:
        array_of_errors.append(e)
    else:
        e = e.mean(0)
        array_of_errors.append(e)
    return array_of_errors

def RMSE(input:tc.tensor, target:tc.tensor, per_time_step: bool = False):
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)
    loss = nn.MSELoss(reduction='none')
    array_of_errors = []
    e = loss(input, target) ** 0.5
    where_to_contract = tuple(np.arange(2,len(input.size()),1))
    if len(input.size())>2:
        e = e.mean(dim = where_to_contract)
    if per_time_step:
        array_of_errors.append(e)
    else:
        e = e.mean(0)
        array_of_errors.append(e)
    return array_of_errors

def RMSE_divided_by_max(input:tc.tensor, target:tc.tensor, per_time_step: bool = False):
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)
    loss = nn.MSELoss(reduction='none')
    array_of_errors = []
    e = loss(input, target) ** 0.5
    where_to_contract = tuple(np.arange(2,len(input.size()),1))
    maxima = tc.amax(e, (0,) + where_to_contract) 
    
    if len(input.size())>2:
        maxima = maxima[None, :, None, None]
        e = e/maxima
        e = e.mean(dim = where_to_contract) 
    
    else:
        maxima = maxima[None, :]
        e = e/maxima
    if per_time_step:
        array_of_errors.append(e)
    else:
        e = e.mean(0)
        array_of_errors.append(e)
    return array_of_errors

def MSE_normalized_by_mean(input:tc.tensor, target:tc.tensor, per_time_step: bool = False):
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)
    loss = nn.MSELoss(reduction='none')
    epsilon = 1e-8
    array_of_errors = []
    e = loss(input, target)
    norm = (target**2)
    where_to_contract = tuple(np.arange(2,len(input.size()),1))
    if len(input.size())>2:
        e = e.mean(dim = where_to_contract)
        norm = norm.mean(dim = where_to_contract)
        
    e_normalized = e/(norm+epsilon)
    if per_time_step:
        array_of_errors.append(e_normalized)
    else:
        e_normalized = e_normalized.mean(0)
        array_of_errors.append(e_normalized)
        
    return array_of_errors

def L2_error_norm(input:tc.tensor, target:tc.tensor, per_time_step: bool = False):
    array_of_errors = []
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)
    epsilon = 1e-8
    if len(input.size()) <= 2: #for scalars it reduces to this computation
        e = tc.abs(input - target).double()
        norm = tc.abs(target).double()
    else:
        e = tc.linalg.vector_norm(input - target, dim = tuple(np.arange(2,len(input.size()),1))).double()
        norm = tc.linalg.vector_norm(target, dim = tuple(np.arange(2,len(input.size()),1))).double()
    e_normalized = e/(norm+epsilon)
    if not per_time_step:
        e_normalized = e_normalized.mean(0)
    array_of_errors.append(e_normalized)
            
    return array_of_errors
    
def compute_global_errors(where_to_get_data:str, where_to_save_data: str, generate_istograms: bool, which_prediction: str):
    txt_files = [f for f in os.listdir(where_to_get_data) if f.endswith('.txt')]
    # create dictionary of errors
    dictionary_of_errors = {
        'MSE' :{},
        'RMSE' :{},
        'MSE_normalized_by_mean' :{},
        'L2_error_norm' : {},
        'RMSE_divided_by_max' :{}
    }
    statistics_errors = {
        'MSE' :{},
        'RMSE' :{},
        'MSE_normalized_by_mean' :{},
        'L2_error_norm' : {},
        'RMSE_divided_by_max' :{}
    }
    
    df = pd.read_csv(f"{where_to_get_data}/{txt_files[0]}", sep='\t')
    for metric in dictionary_of_errors:
        for x in df['Variable name']:
            dictionary_of_errors[metric][x] = []
            statistics_errors[metric][x] = []
        
    # Now cycle over all and append
    for f in txt_files:
        df = pd.read_csv(f"{where_to_get_data}/{f}", sep='\t')
        
        variable_name = df['Variable name']
        
        MSE = df['MSE']
        RMSE = df['RMSE']
        MSE_normalized_by_mean = df['MSE_normalized_by_mean']
        L2_error_norm = df['L2_error_norm']
        RMSE_divided_by_max = df['RMSE_divided_by_max']
        
        for count, name in enumerate(variable_name):
            dictionary_of_errors['MSE'][name].append(MSE[count])
            dictionary_of_errors['RMSE'][name].append(RMSE[count])
            dictionary_of_errors['MSE_normalized_by_mean'][name].append(MSE_normalized_by_mean[count])
            dictionary_of_errors['L2_error_norm'][name].append(L2_error_norm[count])
            dictionary_of_errors['RMSE_divided_by_max'][name].append(RMSE_divided_by_max[count])

            
    # compute average and std for variable
    for metric in dictionary_of_errors:
        for name in dictionary_of_errors[metric]:
            mean = np.mean(dictionary_of_errors[metric][name])
            std = np.std(dictionary_of_errors[metric][name])
            
            statistics_errors[metric][name].append(mean)
            statistics_errors[metric][name].append(std)
            
    for metric in dictionary_of_errors:  
        write_dict_to_txt(statistics_errors[metric], f"{where_to_save_data}/{metric}.txt", len(txt_files), txt_files)

    
    # generate instograms
    if generate_istograms:
        
        for variable in dictionary_of_errors['MSE']:
            plt.figure(figsize = (5,5))
            plt.hist(dictionary_of_errors['MSE'][variable], bins=100, edgecolor='black')
            plt.xlabel('MSE', fontsize = 16)
            plt.ylabel('Frequency', fontsize = 16)
            plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
            plt.savefig(f'{where_to_save_data}/hist_{variable}_MSE.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        for variable in dictionary_of_errors['RMSE']:
            plt.figure(figsize = (5,5))
            plt.hist(dictionary_of_errors['RMSE'][variable], bins=100, edgecolor='black')
            plt.xlabel('RMSE', fontsize = 16)
            plt.ylabel('Frequency', fontsize = 16)
            plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
            plt.savefig(f'{where_to_save_data}/hist_{variable}_RMSE.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        for variable in dictionary_of_errors['MSE_normalized_by_mean']:
            plt.figure(figsize = (5,5))
            plt.hist(dictionary_of_errors['MSE_normalized_by_mean'][variable], bins=100, edgecolor='black')
            plt.xlabel('MSE normalized by mean', fontsize = 16)
            plt.ylabel('Frequency', fontsize = 16)
            plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
            plt.savefig(f'{where_to_save_data}/hist_{variable}_MSE_normalized_by_mean.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        for variable in dictionary_of_errors['L2_error_norm']:
            plt.figure(figsize = (5,5))
            plt.hist(dictionary_of_errors['L2_error_norm'][variable], bins=100, edgecolor='black')
            plt.xlabel('L2 error norm', fontsize = 16)
            plt.ylabel('Frequency', fontsize = 16)
            plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
            plt.savefig(f'{where_to_save_data}/hist_{variable}_L2_error_norm.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        for variable in dictionary_of_errors['RMSE_divided_by_max']:
            plt.figure(figsize = (5,5))
            plt.hist(dictionary_of_errors['RMSE_divided_by_max'][variable], bins=100, edgecolor='black')
            plt.xlabel('RMSE_divided_by_max', fontsize = 16)
            plt.ylabel('Frequency', fontsize = 16)
            plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
            plt.savefig(f'{where_to_save_data}/hist_{variable}_RMSE_divided_by_max.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    return 0

def write_dict_to_txt(data_dict, output_file, how_many_trajectories: int, which_trajectories: list):

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"{'Variable name':<40} {'Mean':<20} {'Std':<20}\n")
        f.write("-" * 80 + "\n")
        
        # Write data rows
        for var_name, values in data_dict.items():
            mean = values[0]
            std = values[1]
            f.write(f"{var_name:<40} {mean:<20.10e} {std:<20.10e}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{how_many_trajectories} total trajectories for testing \n")
        f.write(f"Files used:")
        for i in which_trajectories:
            f.write(f"{i}\n")
    
