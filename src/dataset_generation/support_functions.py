import yaml
import h5py
import numpy as np
import json
import torch as tc
    
def make_faces_array(x): #assumes the input x has dimensions (B,T,C,140)
    shape_x = np.shape(x)
    matrix_of_indeces = np.zeros((16,9))
    matrix_of_indeces[15][0] = 1
    matrix_of_indeces[15][1] = 6
    matrix_of_indeces[15][2] = 2
    matrix_of_indeces[15][3] = 7
    matrix_of_indeces[15][4] = 3
    matrix_of_indeces[15][5] = 8
    matrix_of_indeces[15][6] = 4
    matrix_of_indeces[15][7] = 9
    matrix_of_indeces[15][8] = 5
    
    for i in range(9):
        for k in range(15):
            matrix_of_indeces[14-k][i] = matrix_of_indeces[15][i] + 9 * (k+1)

    reshaped_faces = np.zeros((shape_x[0], shape_x[1],shape_x[2], 16, 9))
    for i in range(16):
        for j in range(9):
            permutation_index = int(matrix_of_indeces[i][j]-1)
            if permutation_index > 139:
                reshaped_faces[:,:,:,i,j] = (x[:,:,:,permutation_index-5] + x[:,:,:,permutation_index-4])/2
            else:
                reshaped_faces[:,:,:,i,j] = x[:,:,:,permutation_index]
        
    return reshaped_faces

def build_dictionary_of_variables():
    dictionary_of_variables = {
        
                'dictionary_of_input_variables_76' : { 
                        'P_vessel': [],
                        'T_gas_vessel': [],
                        'T_liq_vessel': [],
                        'x_alfa_vessel': [], #void fraction
                    
                        'T_sat_vessel': [],
                        'P_H2_vessel': [],
                        'P_steam_vessel': [],
                        'm_gas_vessel': [],
                        'm_liq_vessel_mesh': [],
                        'rho_gas_vessel': [],
                        'rho_liq_vessel': [],
                        'Q_liq_vap_vessel': [],
                        'porosity_vessel': [],
                        'V_deb_vessel': [],
                        'V_mag_vessel': [],
                        
                        'm_magma_vessel': [],
                        'm_debris_0_vessel': [],
                        'm_debris_1_vessel': []
                        },
                        
                        'dictionary_of_input_variables_36' : { 
                            'T_comp_fuel': [],
                            'T_comp_clad': [],
                            
                            'state_fuel': [],
                            'state_clad': []
                        },
                        
                        'dictionary_of_input_variables_1' : { 
                            
                            'm_cum_H2': [],
                            'm_tot_cor': [],
                            'FP_A_heat': [],
                            'sat_core_mesh': []
                        },
                        
                        'dictionary_of_input_variables_140' : {
                            'Q_m_liq_face': [],
                            'V_gas_face': [],
                            'V_liq_face': []
                },
                'vessel_to_primary':{
                    'Q_H20_connection':[],'Q_steam_connection':[], 'm_H20_connection':[], 'time':[] #only place where time information is stored
                },
                'primary_to_vessel':{
                'Q_H20_connection':[],'Q_steam_connection':[], 'm_H20_connection':[]
                }
                                                  }
    return dictionary_of_variables

def fill_dictionary_of_variables(output_dict:dict, name:str, f:h5py._hl.files.File, index_stop:int):
    # === VESSEL GENERAL DATA ===
    # Existing variables
    output_dict[name]['dictionary_of_input_variables_1']['m_cum_H2'].append(np.array(f['vessel/general/m_cum_H2'][:])[0:index_stop])
    output_dict[name]['dictionary_of_input_variables_1']['m_tot_cor'].append(np.array(f['vessel/general/m_tot_cor'][:])[0:index_stop])
    #output_dict[name]['dictionary_of_input_variables_1']['m_tot_deb'].append(np.array(f['vessel/general/m_tot_deb'][:])[0:index_stop]) #it is always nan for some reason
    #output_dict[index_stop]['dictionary_of_input_variables_1']['T_CAVCOR'].append(np.array(f['other/global/sensor_values'][:, 286])[0:index_stop]) #this changes only after vessel rupture
    
    
    # Additional general vessel data
    output_dict[name]['dictionary_of_input_variables_1']['FP_A_heat'].append(np.array(f['vessel/general/FP_A_heat'][:])[0:index_stop])  # Total fission product activity (Bq)
    output_dict[name]['dictionary_of_input_variables_1']['sat_core_mesh'].append(np.array(f['vessel/general/sat_core_mesh'][:])[0:index_stop])  # Maximum saturation in core meshes
    
    # Component temperatures and states (shape: 49095 x 36)
    output_dict[name]['dictionary_of_input_variables_36']['T_comp_fuel'].append(np.array(f['vessel/general/T_comp_fuel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_36']['T_comp_clad'].append(np.array(f['vessel/general/T_comp_clad'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_36']['state_fuel'].append(np.array(f['vessel/general/state_fuel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_36']['state_clad'].append(np.array(f['vessel/general/state_clad'][:])[0:index_stop,:])
    # Debris/magma mass distribution (shape: 49095 x 76)
    output_dict[name]['dictionary_of_input_variables_76']['m_magma_vessel'].append(np.array(f['vessel/general/m_magma_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['m_debris_0_vessel'].append(np.array(f['vessel/general/m_debris_0_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['m_debris_1_vessel'].append(np.array(f['vessel/general/m_debris_1_vessel'][:])[0:index_stop,:])
    
    # === VESSEL MESH DATA (shape: 49095 x 76) ===
    # Thermal properties
    output_dict[name]['dictionary_of_input_variables_76']['P_vessel'].append(np.array(f['vessel/mesh/P_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['T_gas_vessel'].append(np.array(f['vessel/mesh/T_gas_vessel'])[:][0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['T_liq_vessel'].append(np.array(f['vessel/mesh/T_liq_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['T_sat_vessel'].append(np.array(f['vessel/mesh/T_sat_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['x_alfa_vessel'].append(np.array(f['vessel/mesh/x_alfa_vessel'][:])[0:index_stop,:])
    
    # Partial pressures
    output_dict[name]['dictionary_of_input_variables_76']['P_H2_vessel'].append(np.array(f['vessel/mesh/P_H2_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['P_steam_vessel'].append(np.array(f['vessel/mesh/P_steam_vessel'][:])[0:index_stop,:])
    
    # Mass inventories
    output_dict[name]['dictionary_of_input_variables_76']['m_gas_vessel'].append(np.array(f['vessel/mesh/m_gas_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['m_liq_vessel_mesh'].append(np.array(f['vessel/mesh/m_liq_vessel_mesh'][:])[0:index_stop,:])
    
    # Densities
    output_dict[name]['dictionary_of_input_variables_76']['rho_gas_vessel'].append(np.array(f['vessel/mesh/rho_gas_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['rho_liq_vessel'].append(np.array(f['vessel/mesh/rho_liq_vessel'][:])[0:index_stop,:])
    
    # Phase change and geometry
    output_dict[name]['dictionary_of_input_variables_76']['Q_liq_vap_vessel'].append(np.array(f['vessel/mesh/Q_liq_vap_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['porosity_vessel'].append(np.array(f['vessel/mesh/porosity_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['V_deb_vessel'].append(np.array(f['vessel/mesh/V_deb_vessel'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_76']['V_mag_vessel'].append(np.array(f['vessel/mesh/V_mag_vessel'][:])[0:index_stop,:])
    
    # === VESSEL FACE DATA (shape: 49095 x 140) ===
    # Flow rates and velocities
    output_dict[name]['dictionary_of_input_variables_140']['Q_m_liq_face'].append(np.array(f['vessel/face/Q_m_liq_face'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_140']['V_gas_face'].append(np.array(f['vessel/face/V_gas_face'][:])[0:index_stop,:])
    output_dict[name]['dictionary_of_input_variables_140']['V_liq_face'].append(np.array(f['vessel/face/V_liq_face'][:])[0:index_stop,:])

    
    # PRIMARY TO VESSEL (inlet conditions)
    primary_inlet_bc = {
            'Q_H20_connection': f['connection/general/Q_H20_connection'][:, 1],
            'Q_steam_connection': f['connection/general/Q_steam_connection'][:, 1],
            'm_H20_connection': f['connection/general/m_H20_connection'][:, 1],
        }
    
    # VESSEL TO PRIMARY (outlet conditions)
    
    primary_outlet_bc = {
            'Q_H20_connection': f['connection/general/Q_H20_connection'][:, 0],
            'Q_steam_connection': f['connection/general/Q_steam_connection'][:, 0],
            'm_H20_connection': f['connection/general/m_H20_connection'][:, 0],
        }
    
    # === CREATE ORGANIZED BOUNDARY CONDITION DICTIONARY ===
    for i in ['Q_H20_connection','Q_steam_connection', 'm_H20_connection']:
        output_dict[name]['primary_to_vessel'][i].append(np.array(primary_inlet_bc[i])[0:index_stop])
        output_dict[name]['vessel_to_primary'][i].append(np.array(primary_outlet_bc[i])[0:index_stop])
        
    next_time_step = f['dimensions/time_points'][:][1:index_stop]
    previous_time_step = f['dimensions/time_points'][:][0:index_stop-1]
    next_time_step = np.concatenate([next_time_step,[0.0]])
    previous_time_step = np.concatenate([previous_time_step,[0.0]])
    
    DT = next_time_step - previous_time_step
    
    output_dict[name]['vessel_to_primary']['time'].append(DT)

def extract_input_output_bc_variables(path, array_of_datasets:list):
    output_dict = {}
    for i in array_of_datasets:
        with h5py.File(path+'/'+str(i)+'.h5', 'r') as f:
            vessel_rupture_time = f['other/global/vessel_rupture_time'][:][-1]
            if not np.isnan(vessel_rupture_time):
                index_stop = np.where(f['dimensions/time_points'][:] >= vessel_rupture_time)[0][0]
                index_stop = len(f['dimensions/time_points'][:][0:index_stop])
            else:
                index_stop = len(f['dimensions/time_points'][:])
                
            output_dict[i] = build_dictionary_of_variables()
            fill_dictionary_of_variables(output_dict, i, f, index_stop)
                
            
    return output_dict

def dict_to_hdf5(dictionary, h5file, path=''):
    """Recursively save dictionary to HDF5 file."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Create a group for nested dictionaries
            dict_to_hdf5(value, h5file, f"{path}/{key}")
        else:
            # Save the data
            h5file[f"{path}/{key}"] = value
            
def get_normalization_statistics(dictionary_of_sliced_windows:dict, type_of_normalization:str):
    maxima_or_mean = {}
    minima_or_std = {}
    hdf5_keys = list(dictionary_of_sliced_windows.keys())
    if type_of_normalization == 'min_max':
        for key in hdf5_keys:
            shape = np.shape(dictionary_of_sliced_windows[key])
            minimum = np.nanmin(dictionary_of_sliced_windows[key].astype(np.float64),axis = (0,) + tuple(np.arange(2,len(shape))))
            maximum = np.nanmax(dictionary_of_sliced_windows[key].astype(np.float64),axis = (0,) + tuple(np.arange(2,len(shape))))
            minima_or_std[key] = minimum
            maxima_or_mean[key] = maximum
                    
    elif type_of_normalization == 'mean_std':
        for key in hdf5_keys:
            shape = np.shape(dictionary_of_sliced_windows[key])
            mean = np.nanmean(dictionary_of_sliced_windows[key].astype(np.float64), axis=(0,) + tuple(np.arange(2, len(shape))))
            std = np.nanstd(dictionary_of_sliced_windows[key].astype(np.float64), axis=(0,) + tuple(np.arange(2, len(shape))))
            minima_or_std[key] = std
            maxima_or_mean[key] = mean

    else:
        raise TypeError("Type of normalization not known. It can either be min_max or mean_std")                 
    return maxima_or_mean, minima_or_std

def normalize_fields(field: np.array, maximum_or_mean: dict, minimum_or_std: dict, normalization: str):
    size = np.shape(field)
    
    if size[-1] == 7:
        
        maximum_or_mean = maximum_or_mean['boundary_conditions_and_time']
        minimum_or_std = minimum_or_std['boundary_conditions_and_time']
        maximum_or_mean = maximum_or_mean[None,None,:]
        minimum_or_std = minimum_or_std[None,None,:]
        
    elif size[-1] == 4:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_1']
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_1']
        maximum_or_mean = maximum_or_mean[None,None,:]
        minimum_or_std = minimum_or_std[None,None,:]

    elif size[-1] == 140:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_140']
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_140']
        maximum_or_mean = maximum_or_mean[None,None,:, None]
        minimum_or_std = minimum_or_std[None,None,:, None]

    elif size[-1] == 36:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_36']
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_36']
        maximum_or_mean = maximum_or_mean[None,None,:, None]
        minimum_or_std = minimum_or_std[None,None,:, None]

    elif size[-1] == 76:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_76']
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_76']
        maximum_or_mean = maximum_or_mean[None,None,:, None]
        minimum_or_std = minimum_or_std[None,None,:, None]
        
    else:
        raise TypeError(f"Something is wrong with data")
    
    if normalization == 'min_max':
        
        field = ((field - minimum_or_std)/(maximum_or_mean - minimum_or_std))
            
    elif normalization == 'mean_std':
        
        fields = ((field - maximum_or_mean)/minimum_or_std)
        
    elif normalization == 'none':
        return field

    else:
        raise ValueError(f"Missing value") 
        
    return field


