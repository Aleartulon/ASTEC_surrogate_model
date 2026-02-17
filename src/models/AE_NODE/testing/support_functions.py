import yaml
from torch.utils.data import Dataset
import torch as tc
import h5py
import numpy as np
import pandas as pd
import pickle
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
        normalization = tc.mean(np.abs(target), (0,) + where_to_contract) + epsilon
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
    
def compute_global_errors(where_to_get_data:str, string_after_saving:str, where_to_save_data: str, generate_istograms: bool, which_prediction: str):
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
    
    # dictionary of all metrics
    total_dict = {}
        
    #generate single plots with aggregated images
    for metric in dictionary_of_errors:  
        dict_metric = plot_aggregated_errors_per_variable(f"{where_to_save_data}/{metric}.txt", f"{where_to_save_data}",string_after_saving,f"{metric}")
        total_dict[metric] = dict_metric
    
    #generate plots with 3 metrics combined
    combine_metrics_in_one_plot(total_dict, where_to_save_data, string_after_saving)
    
    # generate instograms
    if generate_istograms:
        for metric in dictionary_of_errors:
            for variable in dictionary_of_errors[metric]:
                plt.figure(figsize = (5,5))
                plt.hist(dictionary_of_errors[metric][variable], bins=100, edgecolor='black')
                plt.xlabel(metric, fontsize = 16)
                plt.ylabel('Frequency', fontsize = 16)
                plt.title(f"{variable.replace('_', ' ')}, {which_prediction} prediction", fontsize = 16)
                plt.savefig(f'{where_to_save_data}/hist_{variable}_{metric}_{string_after_saving}.png', dpi=300, bbox_inches='tight')
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
            
def plot_aggregated_errors_per_variable(path_txt_file:str, saving_path:str,string_after_saving:str, metric:str):
    df = pd.read_fwf(path_txt_file, skiprows=2, skipfooter=1, engine='python',
                  colspecs=[(0, 45), (45, 65), (66, 86)],
                  names=['Variable', 'Mean', 'Std'])
    dict_global = {}
    dict_vessel = {}
    dict_plenum = {}
    dict_core = {}
    dict_faces = {}
    
    to_be_skipped_RMSE_divided_by_max = ['Q_fp_Ac_scalar', 'Q_fp_Pa_scalar', 'Q_fp_Ra_scalar', 'Q_fp_Re_scalar', 'Q_fp_Th_scalar',  'Q_fp_Tl_scalar', 'm_debris_0_lower_plenum', 'm_debris_1_lower_plenum', 'm_magma_lower_plenum', 'm_debris_0_lower_vessel_vessel','m_debris_1_lower_vessel_vessel','m_magma_vessel_vessel'] #they are constant
    to_be_skipped_RMSE_divided_by_mean = ['Q_fp_Ac_scalar', 'Q_fp_Pa_scalar', 'Q_fp_Ra_scalar', 'Q_fp_Re_scalar', 'Q_fp_Th_scalar',  'Q_fp_Tl_scalar', 'm_debris_0_lower_plenum', 'm_debris_1_lower_plenum', 'm_magma_lower_plenum', 'm_debris_0_lower_vessel_vessel','m_debris_1_lower_vessel_vessel','m_magma_vessel_vessel'] #they are constant
    to_be_skipped_RMSE_divided_by_std = ['Q_fp_Ac_scalar', 'Q_fp_Pa_scalar', 'Q_fp_Ra_scalar', 'Q_fp_Re_scalar', 'Q_fp_Th_scalar',  'Q_fp_Tl_scalar', 'm_debris_0_lower_plenum', 'm_debris_1_lower_plenum', 'm_magma_lower_plenum', 'm_debris_0_lower_vessel_vessel','m_debris_1_lower_vessel_vessel','m_magma_vessel_vessel'] #they are constant
    
    for count, i in enumerate(df['Variable']):
        if metric == 'RMSE_divided_by_max':
            if i in to_be_skipped_RMSE_divided_by_max:
                continue
        elif metric == 'RMSE_divided_by_mean':
            if i in to_be_skipped_RMSE_divided_by_mean:
                continue
        elif metric == 'RMSE_divided_by_std':
            if i in to_be_skipped_RMSE_divided_by_std:
                continue
        
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

    dict_global = make_hist_and_plots(dict_global, saving_path,string_after_saving ,metric, latex_metric = which_metric(metric, "g"), label = 'g')
    dict_vessel = make_hist_and_plots(dict_vessel, saving_path ,string_after_saving,metric,  latex_metric = which_metric(metric, "v"), label = 'v',)
    dict_plenum = make_hist_and_plots(dict_plenum, saving_path,string_after_saving ,metric,  latex_metric = which_metric(metric, "p"), label = 'p')
    dict_core = make_hist_and_plots(dict_core, saving_path,string_after_saving ,metric, latex_metric = which_metric(metric, "cr"), label = 'cr')
    dict_faces = make_hist_and_plots(dict_faces, saving_path,string_after_saving,metric, latex_metric = which_metric(metric, "f"), label = 'f')
    
    dict_metric = { 'g': dict_global, 'v' : dict_vessel, 'p' :dict_plenum,'cr': dict_core, 'f': dict_faces}
    return dict_metric

def make_hist_and_plots(data_dictionary:dict, path:str,string_after_saving:str,metric:str, latex_metric:str, label:str):
    
    #aggregate fission products for ease of representation
    if label == 'g':
        new_dict = {}
        mean = 0
        std = 0
        count = 0
        for i in data_dictionary:
            if i[:4] == 'Q fp':
                mean += data_dictionary[i][0]
                std += data_dictionary[i][1]
                count+=1
            else:
                new_dict[i] = data_dictionary[i]
        new_dict['FP'] = [mean/count, std/count]
        data_dictionary = new_dict
        
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
    plt.savefig(f'{path}/{metric}/{metric}_{label}_histogram_{string_after_saving}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    #make plots 
    plot_array_x = [x for x in data_dictionary]
    plot_array_y = [data_dictionary[x][0] for x in data_dictionary]
    plot_array_unc = [data_dictionary[x][1] for x in data_dictionary]
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    x_pos = range(len(plot_array_x))
    axs.errorbar(x_pos, plot_array_y, yerr=plot_array_unc, fmt='o', capsize=5)
    axs.set_xticks(x_pos)
    axs.set_xticklabels(plot_array_x, rotation=90, ha='right')
    axs.set_ylabel(latex_metric, fontsize=16)
    if metric == "RMSE_divided_by_std":
        axs.hlines(0.5, xmin=0, xmax=len(plot_array_x)-1, colors='green', linestyles='solid')
    axs.set_yscale('log')
    plt.title(rf'{latex_metric} for $s_{{{label}}}$ variables', fontsize=16)
    os.makedirs(f'{path}/{metric}/', exist_ok=True)
    plt.savefig(f'{path}/{metric}/{metric}_{label}_plot_{string_after_saving}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return data_dictionary
    
    
    
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
    
def combine_metrics_in_one_plot(total_dict:dict, where_to_save_data:str, string_after_saving:str):
    metrics_to_be_plotted = ['RMSE_divided_by_mean', 'RMSE_divided_by_max', 'RMSE_divided_by_std']
    variables_to_be_plotted = ['g','v','p','cr','f']
    colors = ['blue', 'red', 'green']
    
    os.makedirs(f'{where_to_save_data}/combined_plots/', exist_ok=True)
    
    # Dictionary to store plotting data
    plotting_data = {}
    
    for variable in variables_to_be_plotted:
        plotting_data[variable] = {}
        if variable == 'g':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (12,5))
        elif variable == 'v':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (12,5))
        elif variable == 'p':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (12,5))
        elif variable == 'cr':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (5,5))
        elif variable == 'f':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (5,5))
        
        for idx, metric in enumerate(metrics_to_be_plotted):
            plot_array_x = [x for x in total_dict[metric][variable]]
            plot_array_y = [total_dict[metric][variable][x][0] for x in total_dict[metric][variable]]
            plot_array_unc = [total_dict[metric][variable][x][1] for x in total_dict[metric][variable]]
            
            for count, i in enumerate(plot_array_x):
                split = i.rsplit(' ', 1)
                if i =='Q H20 connection primary to vessel':
                    plot_array_x[count] = 'Q H20 ptv'
                elif i =='Q H20 connection vessel to primary':
                    plot_array_x[count] = 'Q H20 vtp'
                elif i =='Q steam connection vessel to primary':
                    plot_array_x[count] = 'Q steam vtp'
                elif i =='Q steam connection primary to vessel':
                    plot_array_x[count] = 'Q steam ptv'
                elif i =='m H20 connection vessel to primary':
                    plot_array_x[count] = 'm H20 vtp'
                elif i =='m H20 connection primary to vessel':
                    plot_array_x[count] = 'm H20 ptv'
                elif i =='m liq vessel mesh':
                    plot_array_x[count] = 'm liq'
                elif split[-1] == 'face' or split[-1] == 'vessel' or split[-1] == 'lower':
                    plot_array_x[count] = split[0]
            
            # Store the processed data
            plotting_data[variable][metric] = {
                'labels': plot_array_x.copy(),  # Use .copy() to avoid reference issues
                'values': plot_array_y,
                'uncertainties': plot_array_unc
            }
            
            x_pos = range(len(plot_array_x))
            
            if metric == 'RMSE_divided_by_mean':
                label = r'RMSE$_{mean}$'
            elif metric == 'RMSE_divided_by_max':
                label = r'RMSE$_{max}$'
            elif metric == 'RMSE_divided_by_std':
                label = r'RMSE$_{std}$'
            
            axs.errorbar(x_pos, plot_array_y, yerr=plot_array_unc, 
                        fmt='o', capsize=5, 
                        color=colors[idx],
                        label=label)
        
        axs.set_ylabel('Error', fontsize=16)
        axs.hlines(0.5, xmin=0, xmax=len(plot_array_x)-1, colors='green', linestyles='dashed')
        axs.set_yscale('log')
        axs.set_xticks(x_pos)
        axs.set_xticklabels([])
        axs.tick_params(axis='y', labelsize=16) 
        
        for i, (pos, label_text) in enumerate(zip(x_pos, plot_array_x)):
            y_offset = -0.02 if i % 2 == 0 else -0.06
            axs.text(pos, y_offset, label_text, 
                    ha='center', 
                    va='top',
                    transform=axs.get_xaxis_transform(),
                    fontsize=10, rotation=45)
        
        if variable == 'g':
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        axs.set_title(rf'$s_{{{variable}}}$', fontsize =16)
        plt.savefig(f'{where_to_save_data}/combined_plots/{variable}_{string_after_saving}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Save the plotting data
    
    with open(f'{where_to_save_data}/combined_plots/plotting_data.pkl', 'wb') as f:
        pickle.dump(plotting_data, f)
    
    return 0
def compare_errors_AE_and_AE_NODE(path_AE:str, path_AE_NODE:str, where_to_save:str, string_after_saving:str):
    import pickle
    
    # Load the data from both models
    with open(f'{path_AE}/combined_plots/plotting_data.pkl', 'rb') as f:
        AE_data = pickle.load(f)
    
    with open(f'{path_AE_NODE}/combined_plots/plotting_data.pkl', 'rb') as f:
        AE_NODE_data = pickle.load(f)
    
    # Define what to plot
    metrics_to_be_plotted = ['RMSE_divided_by_mean', 'RMSE_divided_by_max', 'RMSE_divided_by_std']
    variables_to_be_plotted = ['g','v','p','cr','f']
    colors = ['blue', 'red', 'green']
    
    # Create directory for comparison plots
    os.makedirs(f'{where_to_save}/comparison_plots/', exist_ok=True)
    
    for variable in variables_to_be_plotted:
        if variable == 'g':
            fig, axs = plt.subplots(1, 1, tight_layout=True,figsize = (10,5))
        else:
            fig, axs = plt.subplots(1, 1, tight_layout=True)
        
        for idx, metric in enumerate(metrics_to_be_plotted):
            # Get data from both models
            AE_metric = AE_data[variable][metric]
            AE_NODE_metric = AE_NODE_data[variable][metric]
            
            labels = AE_metric['labels']  # Should be same for both
            x_pos = range(len(labels))
            
            # Metric label
            if metric == 'RMSE_divided_by_mean':
                label_base = r'RMSE$_{mean}$'
            elif metric == 'RMSE_divided_by_max':
                label_base = r'RMSE$_{max}$'
            elif metric == 'RMSE_divided_by_std':
                label_base = r'RMSE$_{std}$'
            
            # Plot AE (circles)
            axs.errorbar(x_pos, AE_metric['values'], yerr=AE_metric['uncertainties'], 
                        fmt='o', capsize=5, 
                        color=colors[idx],
                        label=f'{label_base} AE',
                        alpha=0.7)
            
            # Plot AE_NODE (squares)
            axs.errorbar(x_pos, AE_NODE_metric['values'], yerr=AE_NODE_metric['uncertainties'], 
                        fmt='s', capsize=5, 
                        color=colors[idx],
                        label=f'{label_base} AE-NODE',
                        alpha=0.7)
        
        axs.set_ylabel('Error', fontsize=16)
        axs.hlines(0.5, xmin=0, xmax=len(labels)-1, colors='green', linestyles='dashed')
        axs.set_yscale('log')
        axs.set_xticks(x_pos)
        axs.set_xticklabels([])
        axs.tick_params(axis='y', labelsize=16) 
        
        # Manually place labels with alternating heights
        for i, (pos, label_text) in enumerate(zip(x_pos, labels)):
            y_offset = -0.02 if i % 2 == 0 else -0.06
            axs.text(pos, y_offset, label_text, 
                    ha='center', 
                    va='top',
                    transform=axs.get_xaxis_transform(),
                    fontsize=10, rotation=45)
        
        if variable == 'g':
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        axs.set_title(rf'$s_{{{variable}}}$ - AE vs AE-NODE comparison', fontsize = 16)
        
        plt.savefig(f'{where_to_save}/comparison_plots/{variable}_comparison_{string_after_saving}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return 0