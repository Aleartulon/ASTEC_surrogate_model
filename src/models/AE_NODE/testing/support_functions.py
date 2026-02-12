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
    def __init__(self, training_config:dict, f: F_Latent, device:tc.device, substep_RK4:int):
        # Required for processor_First_Order
        self.k = training_config['k']  # RK order (1 for Euler method)
        self.device = device
        self.f = f  # The dynamics function f(latent, boundaries)
        self.RK = {k: tc.tensor([[safe_eval(val) for val in row] for row in v]) for k, v in training_config['RK'].items()}
        self.substep_RK4 = substep_RK4
        
def safe_eval( val):
    if isinstance(val, str):
        return eval(val)
    return val

def compute_errors(trajectory:str, input: list, target:list, is_AE:bool): #WATCH OUT, convention is that the ones in time always end with step, look at generate_pictures_errors_field_reconstruction in model_test.py
    dictionary_of_errors = {}
    dictionary_of_errors[str(trajectory)] = {'MSE' : [],
                                             'RMSE' : [],
                                            'MSE_normalized_by_mean':[], 
                                             'L2_error_norm' : [], 
                                             'RMSE_divided_by_max' : [],
                                             'RMSE_divided_by_mean' : [],
                                             'RMSE_divided_by_std' : [],
                                             
                                             'MSE_per_time_step':[], 
                                             'RMSE_per_time_step':[], 
                                             'MSE_normalized_by_mean_per_time_step':[], 
                                             'L2_error_norm_per_time_step' : [],
                                             'RMSE_divided_by_max_per_time_step':[],
                                             'RMSE_divided_by_mean_per_time_step' : [],
                                             'RMSE_divided_by_std_per_time_step' : []
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
        error = RMSE_divided_by_something(i, target[count], 'max')
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_max'].append(error)
        
        #compute RMSE_divided_by_mean
        error = RMSE_divided_by_something(i, target[count], 'mean')
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_mean'].append(error)
        
        #compute RMSE_divided_by_std
        error = RMSE_divided_by_something(i, target[count], 'std')
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_std'].append(error)
        
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
        error = RMSE_divided_by_something(i, target[count], 'max', per_time_step = True)
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_max_per_time_step'].append(error)
        
        #compute RMSE_divided_by_mean_per_time_step
        error = RMSE_divided_by_something(i, target[count], 'mean', per_time_step = True)
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_mean_per_time_step'].append(error)
        
        #compute RMSE_divided_by_std_per_time_step
        error = RMSE_divided_by_something(i, target[count], 'std', per_time_step = True)
        dictionary_of_errors[str(trajectory)]['RMSE_divided_by_std_per_time_step'].append(error)
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
    e = loss(input, target) 
    if len(input.size()) >2:
        where_to_contract = tuple(np.arange(2,len(input.size()),1))
        e = e.mean(dim = where_to_contract)** 0.5
    else:
        e = e** 0.5
    if per_time_step:
        array_of_errors.append(e)
    else:
        e = e.mean(0)
        array_of_errors.append(e)
    return array_of_errors

def RMSE_divided_by_something(input:tc.tensor, target:tc.tensor, which_division:str, per_time_step: bool = False):
    epsilon = 1e-8
    input = input.double().squeeze(0)
    target = target.double().squeeze(0)

    loss = nn.MSELoss(reduction='none')
    array_of_errors = []
    e = loss(input, target)
    where_to_contract = tuple(np.arange(2,len(input.size()),1))
    if which_division == 'max':
        normalization = tc.amax(target, (0,) + where_to_contract) + epsilon
    elif which_division == 'mean':
        normalization = tc.mean(tc.abs(target), (0,) + where_to_contract) + epsilon
    elif which_division == 'std':
        normalization = tc.std(target, (0,) + where_to_contract) + epsilon
    else:
        raise TypeError('Division you put is not existent.')
    
    if len(input.size())>2:
        e = e.mean(dim = where_to_contract)**0.5
        e = e/normalization
    else:
        e = e**0.5
        e = e/normalization
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
    df = pd.read_csv(f"{where_to_get_data}/{txt_files[0]}", sep='\t')
    # create dictionary of errors
    dictionary_of_errors = {
    }
    statistics_errors = {
    }
    
    for metric in df.keys()[1:-1]:
        dictionary_of_errors[metric] = {}
        statistics_errors[metric] = {}
        
    df = pd.read_csv(f"{where_to_get_data}/{txt_files[0]}", sep='\t')
    for metric in dictionary_of_errors:
        for x in df[df.columns[0]]:
            dictionary_of_errors[metric][x] = []
            statistics_errors[metric][x] = []
        
    # Now cycle over all and append
    for f in txt_files:
        df = pd.read_csv(f"{where_to_get_data}/{f}", sep='\t')
        
        variable_name = df[df.columns[0]]
        for count, name in enumerate(variable_name):
            for metric in dictionary_of_errors:
                column = df[metric]
                dictionary_of_errors[metric][name].append(column[count])
            
    # compute average and std for variable
    for metric in dictionary_of_errors:
        for name in dictionary_of_errors[metric]:
            mean = np.mean(dictionary_of_errors[metric][name])
            std = np.std(dictionary_of_errors[metric][name])
            
            statistics_errors[metric][name].append(mean)
            statistics_errors[metric][name].append(std)
            
    for metric in dictionary_of_errors:  
        write_dict_to_txt(statistics_errors[metric], f"{where_to_save_data}/{metric}.txt")
    
    write_info(f"{where_to_save_data}/info.txt", len(txt_files), txt_files)
        
    #generate plots with aggregated images
    for metric in dictionary_of_errors:  
        plot_aggregated_errors_per_variable(f"{where_to_save_data}/{metric}.txt", f"{where_to_save_data}",f"{metric}")

    # generate instograms
    if generate_istograms:
        for metric in dictionary_of_errors:
            for variable in dictionary_of_errors[metric]:
                plt.figure(figsize = (5,5))
                plt.hist(dictionary_of_errors[metric][variable], bins=100, edgecolor='black')
                plt.xlabel(metric, fontsize = 16)
                plt.ylabel('Frequency', fontsize = 16)
                plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
                plt.savefig(f'{where_to_save_data}/hist_{variable}_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
    
    return 0

def write_dict_to_txt(data_dict, output_file):

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"{'Variable name':<44} {'Mean':<20} {'Std':<20}\n")
        f.write("-" * 80 + "\n")
        
        # Write data rows
        for var_name, values in data_dict.items():
            mean = values[0]
            std = values[1]
            f.write(f"{var_name:<44} {mean:<20.10e} {std:<20.10e}\n")
        f.write("-" * 80 + "\n")

def write_info( output_file, how_many_trajectories: int, which_trajectories: list):

    with open(output_file, 'w') as f:
        f.write(f"{how_many_trajectories} total trajectories for testing \n")
        f.write(f"Files used:\n")
        for i in which_trajectories:
            f.write(f"{i}\n")
            
def plot_aggregated_errors_per_variable(path_txt_file:str, saving_path:str, metric:str):
    df = pd.read_fwf(path_txt_file, skiprows=2, skipfooter=1, engine='python',
                  colspecs=[(0, 45), (45, 64), (64, 85)],
                  names=['Variable', 'Mean', 'Std'])
    dict_global = {}
    dict_vessel = {}
    dict_plenum = {}
    dict_core = {}
    dict_faces = {}
    
    for count, i in enumerate(df['Variable']):
        split = i.rsplit('_', 1)
        name = split[0].replace('_', ' ') 
        if split[-1] == 'scalar':
            dict_global[name] = [df['Mean'][count],df['Std'][count]]
        elif split[-1] == 'vessel':
            dict_vessel[name] = [df['Mean'][count],df['Std'][count]]
        elif split[-1] == 'core':
            dict_core[name] = [df['Mean'][count],df['Std'][count]]
        elif split[-1] == 'faces':
            dict_faces[name] = [df['Mean'][count],df['Std'][count]]
        elif split[-1] == 'plenum':
            dict_plenum[name] = [df['Mean'][count],df['Std'][count]]
        else:
            raise TypeError('Something wrong')
    
    
        
    make_hist_and_plots(dict_global, saving_path ,metric, latex_metric = which_metric(metric, "g"), label = 'g')
    make_hist_and_plots(dict_vessel, saving_path ,metric,  latex_metric = which_metric(metric, "v"), label = 'v')
    make_hist_and_plots(dict_plenum, saving_path ,metric,  latex_metric = which_metric(metric, "p"), label = 'p')
    make_hist_and_plots(dict_core, saving_path ,metric, latex_metric = which_metric(metric, "cr"), label = 'cr')
    make_hist_and_plots(dict_faces, saving_path ,metric, latex_metric = which_metric(metric, "f"), label = 'f')
    
    return 0

def make_hist_and_plots(data_dictionary:dict, path:str,metric:str, latex_metric:str, label:str):
    
    #make histograms
    
    hist_array = [data_dictionary[x][0] for x in data_dictionary]
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(hist_array, bins=np.logspace(np.log10(min(hist_array)), np.log10(max(hist_array)), 100))
    axs.set_xscale('log')
    axs.set_xlabel(latex_metric, fontsize = 16)
    axs.set_ylabel('Frequency', fontsize = 16)
    median_val = np.median(hist_array)
    axs.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2e}')
    plt.title(rf'{latex_metric} distribution for $s_{{{label}}}$ variables', fontsize=16)
    os.makedirs(f'{path}/{metric}/', exist_ok=True)
    plt.savefig(f'{path}/{metric}/{metric}_{label}_histogram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    #make plots
    
def which_metric(metric, variable):
    if metric == "RMSE":
        return rf"$RMSE_{{{variable}}}$"
    elif metric == "RMSE_divided_by_mean":
        return rf"$RMSE_{{{variable}, mean}}$"
    elif metric == "RMSE_divided_by_std":
        return rf"$RMSE_{{{variable}, std}}$"
    elif metric == "RMSE_divided_by_max":
        return rf"$RMSE_{{{variable}, max}}$"
    elif metric == "MSE":
        return rf"$MSE_{{{variable}}}$"
    elif metric == "L2_error_norm":
        return rf"$L2_error_norm_{{{variable}}}$"
    elif metric == "MSE_normalized_by_mean":
        return rf"$MSE normalized by mean_{{{variable}}}$"
    else:
        raise TypeError(f"Something wrong, metric is {metric}")
    