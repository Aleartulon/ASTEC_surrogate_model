import yaml
import h5py
import numpy as np
import json
import torch as tc
import math
    
def make_faces_array(x, device:tc.device):
    # Pre-compute index matrix (could even be done outside function as a constant)
    base_indices = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5])
    offsets = np.arange(15, -1, -1) * 9  # [135, 126, ..., 9, 0]
    matrix_of_indices = tc.tensor(base_indices[np.newaxis, :] + offsets[:, np.newaxis], device = device)
    
    # Convert to 0-indexed
    matrix_of_indices = matrix_of_indices - 1
    
    # Identify which indices need averaging
    needs_averaging = matrix_of_indices > 139
    
    # Create output array
    shape_x = x.shape
    reshaped_faces = tc.zeros((shape_x[0], shape_x[1], 16, 9), device = device)
    
    # Vectorized assignment for normal indices
    normal_mask = ~needs_averaging
    normal_indices = matrix_of_indices[normal_mask]
    i_coords, j_coords = tc.where(normal_mask)
    reshaped_faces[ :, :, i_coords, j_coords] = x[ :, :, normal_indices]
    
    # Vectorized assignment for averaged indices
    avg_indices = matrix_of_indices[needs_averaging]
    i_coords, j_coords = tc.where(needs_averaging)
    reshaped_faces[ :, :, i_coords, j_coords] = (
        x[ :, :, avg_indices - 5] + x[ :, :, avg_indices - 4]
    ) / 2
    
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
                    'Q_H20_connection':[],'Q_steam_connection':[], 'm_H20_connection':[], 'dt':[] ,'time':[] #only place where time information is stored
                },
                'VDO':{ 'T_gas_primary_volume': [],'x_alfa_primary_volume':[],'P_steam_primary_volume': [],'P_saturation_primary_volume': [],'P_H2_primary_volume': [],'P_primary_volume': [],
                        'm_steam_primary_volume': [],'rho_liq_primary_volume':[], 'm_liq_primary_volume': [], 'T_sat_primary_volume': [],'x_steam_primary_volume': [],'T_liq_primary_volume': []
                    },
                'UPP_V001':{ 'T_gas_primary_volume': [],'x_alfa_primary_volume':[],'P_steam_primary_volume': [],'P_saturation_primary_volume': [],'P_H2_primary_volume': [],'P_primary_volume': [],
                        'm_steam_primary_volume': [],'rho_liq_primary_volume':[], 'm_liq_primary_volume': [], 'T_sat_primary_volume': [],'x_steam_primary_volume': [],'T_liq_primary_volume': []
                    },
                'primary_to_vessel':{
                'Q_H20_connection':[],'Q_steam_connection':[], 'm_H20_connection':[]
                }
                                                  }
    return dictionary_of_variables

def fill_dictionary_of_variables(output_dict:dict, name:str, f:h5py._hl.files.File, index_stop:int, subsampling_index:int):
    # === VESSEL GENERAL DATA ===
    # Existing variables
    output_dict[name]['dictionary_of_input_variables_1']['m_cum_H2'] = f['vessel/general/m_cum_H2'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['m_tot_cor'] = f['vessel/general/m_tot_cor'][0:index_stop][::subsampling_index]
    #output_dict[name]['dictionary_of_input_variables_1']['m_tot_deb'].append(np.array(f['vessel/general/m_tot_deb'][:])[0:index_stop][::subsampling_index]) #it is always nan for some reason
    #output_dict[index_stop]['dictionary_of_input_variables_1']['T_CAVCOR'].append(np.array(f['other/global/sensor_values'][:, 286])[0:index_stop][::subsampling_index]) #this changes only after vessel rupture
    
    
    # Additional general vessel data
    output_dict[name]['dictionary_of_input_variables_1']['FP_A_heat'] = f['vessel/general/FP_A_heat'][0:index_stop][::subsampling_index] # Total fission product activity (Bq)
    output_dict[name]['dictionary_of_input_variables_1']['sat_core_mesh']=f['vessel/general/sat_core_mesh'][0:index_stop][::subsampling_index] # Maximum saturation in core meshes
    
    # Fission products
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ac'] = f['connection/fission/Q_fp_Ac'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ag'] = f['connection/fission/Q_fp_Ag'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Am'] = f['connection/fission/Q_fp_Am'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_As'] = f['connection/fission/Q_fp_As'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ba'] = f['connection/fission/Q_fp_Ba'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Br'] = f['connection/fission/Q_fp_Br'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Cd'] = f['connection/fission/Q_fp_Cd'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ce'] = f['connection/fission/Q_fp_Ce'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Cm'] = f['connection/fission/Q_fp_Cm'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Cs'] = f['connection/fission/Q_fp_Cs'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Cu'] = f['connection/fission/Q_fp_Cu'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Dy'] = f['connection/fission/Q_fp_Dy'][0:index_stop][::subsampling_index]
    
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Er'] = f['connection/fission/Q_fp_Er'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Eu'] = f['connection/fission/Q_fp_Eu'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ga'] = f['connection/fission/Q_fp_Ga'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Gd'] = f['connection/fission/Q_fp_Gd'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ge'] = f['connection/fission/Q_fp_Ge'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ho'] = f['connection/fission/Q_fp_Ho'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_I'] = f['connection/fission/Q_fp_I'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_In'] = f['connection/fission/Q_fp_In'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Kr'] = f['connection/fission/Q_fp_Kr'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_La'] = f['connection/fission/Q_fp_La'][0:index_stop][::subsampling_index]
    
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Mo'] = f['connection/fission/Q_fp_Mo'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Nb'] = f['connection/fission/Q_fp_Nb'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Nd'] = f['connection/fission/Q_fp_Nd'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Np'] = f['connection/fission/Q_fp_Np'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Pa'] = f['connection/fission/Q_fp_Pa'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Pd'] = f['connection/fission/Q_fp_Pd'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Pm'] = f['connection/fission/Q_fp_Pm'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Pr'] = f['connection/fission/Q_fp_Pr'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Pu'] = f['connection/fission/Q_fp_Pu'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ra'] = f['connection/fission/Q_fp_Ra'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Rb'] = f['connection/fission/Q_fp_Rb'][0:index_stop][::subsampling_index]
    
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Re'] = f['connection/fission/Q_fp_Re'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Rh'] = f['connection/fission/Q_fp_Rh'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Ru'] = f['connection/fission/Q_fp_Ru'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Sb'] = f['connection/fission/Q_fp_Sb'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Se'] = f['connection/fission/Q_fp_Se'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Sm'] = f['connection/fission/Q_fp_Sm'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Sn'] = f['connection/fission/Q_fp_Sn'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Sr'] = f['connection/fission/Q_fp_Sr'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Tb'] = f['connection/fission/Q_fp_Tb'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Tc'] = f['connection/fission/Q_fp_Tc'][0:index_stop][::subsampling_index]
    
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Te'] = f['connection/fission/Q_fp_Te'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Th'] = f['connection/fission/Q_fp_Th'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Tl'] = f['connection/fission/Q_fp_Tl'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Tm'] = f['connection/fission/Q_fp_Tm'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_U'] = f['connection/fission/Q_fp_U'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Xe'] = f['connection/fission/Q_fp_Xe'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Y'] = f['connection/fission/Q_fp_Y'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Yb'] = f['connection/fission/Q_fp_Yb'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Zn'] = f['connection/fission/Q_fp_Zn'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_1']['Q_fp_Zr'] = f['connection/fission/Q_fp_Zr'][0:index_stop][::subsampling_index]
    
    # Component temperatures and states (shape: 49095 x 36)
    output_dict[name]['dictionary_of_input_variables_36']['T_comp_fuel']=f['vessel/general/T_comp_fuel'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_36']['T_comp_clad']=f['vessel/general/T_comp_clad'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_36']['state_fuel']=f['vessel/general/state_fuel'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_36']['state_clad']=f['vessel/general/state_clad'][0:index_stop][::subsampling_index]
    # Debris/magma mass distribution (shape: 49095 x 76)
    output_dict[name]['dictionary_of_input_variables_76']['m_magma_vessel']=f['vessel/general/m_magma_vessel'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_76']['m_debris_0_vessel']=f['vessel/general/m_debris_0_vessel'][0:index_stop][::subsampling_index]
    output_dict[name]['dictionary_of_input_variables_76']['m_debris_1_vessel']=f['vessel/general/m_debris_1_vessel'][:index_stop, :][::subsampling_index, :]
    
    # === VESSEL MESH DATA (shape: 49095 x 76) ===
    # Thermal properties
    output_dict[name]['dictionary_of_input_variables_76']['P_vessel']=f['vessel/mesh/P_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['T_gas_vessel']=f['vessel/mesh/T_gas_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['T_liq_vessel']=f['vessel/mesh/T_liq_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['T_sat_vessel']=f['vessel/mesh/T_sat_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['x_alfa_vessel']=f['vessel/mesh/x_alfa_vessel'][:index_stop, :][::subsampling_index, :]
    
    # Partial pressures
    output_dict[name]['dictionary_of_input_variables_76']['P_H2_vessel']=f['vessel/mesh/P_H2_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['P_steam_vessel']=f['vessel/mesh/P_steam_vessel'][:index_stop, :][::subsampling_index, :]
    
    # Mass inventories
    output_dict[name]['dictionary_of_input_variables_76']['m_gas_vessel']=f['vessel/mesh/m_gas_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['m_liq_vessel_mesh']=f['vessel/mesh/m_liq_vessel_mesh'][:index_stop, :][::subsampling_index, :]
    
    # Densities
    output_dict[name]['dictionary_of_input_variables_76']['rho_gas_vessel']=f['vessel/mesh/rho_gas_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['rho_liq_vessel']=f['vessel/mesh/rho_liq_vessel'][:index_stop, :][::subsampling_index, :]
    
    # Phase change and geometry
    output_dict[name]['dictionary_of_input_variables_76']['Q_liq_vap_vessel']=f['vessel/mesh/Q_liq_vap_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['porosity_vessel']=f['vessel/mesh/porosity_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['V_deb_vessel']=f['vessel/mesh/V_deb_vessel'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_76']['V_mag_vessel']=f['vessel/mesh/V_mag_vessel'][:index_stop, :][::subsampling_index, :]
    
    # === VESSEL FACE DATA (shape: 49095 x 140) ===
    # Flow rates and velocities
    output_dict[name]['dictionary_of_input_variables_140']['Q_m_liq_face']=f['vessel/face/Q_m_liq_face'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_140']['V_gas_face']=f['vessel/face/V_gas_face'][:index_stop, :][::subsampling_index, :]
    output_dict[name]['dictionary_of_input_variables_140']['V_liq_face']=f['vessel/face/V_liq_face'][:index_stop, :][::subsampling_index, :]

    
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
    
    # PROMARY VOLUME VDO
    VDO = {'T_gas_primary_volume': f['primary/volume/T_gas_primary_volume'][:,0],
           'x_alfa_primary_volume': f['primary/volume/x_alfa_primary_volume'][:,0],
           'P_steam_primary_volume': f['primary/volume/P_steam_primary_volume'][:,0],
           'P_saturation_primary_volume': f['primary/volume/P_saturation_primary_volume'][:,0],
           'P_H2_primary_volume': f['primary/volume/P_H2_primary_volume'][:,0],
           'P_primary_volume': f['primary/volume/P_primary_volume'][:,0],
           'm_steam_primary_volume': f['primary/volume/m_steam_primary_volume'][:,0],
           'rho_liq_primary_volume': f['primary/volume/rho_liq_primary_volume'][:,0],
           'm_liq_primary_volume': f['primary/volume/m_liq_primary_volume'][:,0],
           'T_sat_primary_volume': f['primary/volume/T_sat_primary_volume'][:,0],
           'x_steam_primary_volume': f['primary/volume/x_steam_primary_volume'][:,0],
           'T_liq_primary_volume': f['primary/volume/T_liq_primary_volume'][:,0],
           }
    
    UPP_V001 = {'T_gas_primary_volume': f['primary/volume/T_gas_primary_volume'][:,12],
           'x_alfa_primary_volume': f['primary/volume/x_alfa_primary_volume'][:,12],
           'P_steam_primary_volume': f['primary/volume/P_steam_primary_volume'][:,12],
           'P_saturation_primary_volume': f['primary/volume/P_saturation_primary_volume'][:,12],
           'P_H2_primary_volume': f['primary/volume/P_H2_primary_volume'][:,12],
           'P_primary_volume': f['primary/volume/P_primary_volume'][:,12],
           'm_steam_primary_volume': f['primary/volume/m_steam_primary_volume'][:,12],
           'rho_liq_primary_volume': f['primary/volume/rho_liq_primary_volume'][:,12],
           'm_liq_primary_volume': f['primary/volume/m_liq_primary_volume'][:,12],
           'T_sat_primary_volume': f['primary/volume/T_sat_primary_volume'][:,12],
           'x_steam_primary_volume': f['primary/volume/x_steam_primary_volume'][:,12],
           'T_liq_primary_volume': f['primary/volume/T_liq_primary_volume'][:,12],
           }
    
    # === CREATE ORGANIZED BOUNDARY CONDITION DICTIONARY ===
    for i in ['Q_H20_connection','Q_steam_connection', 'm_H20_connection']:
        output_dict[name]['primary_to_vessel'][i]=np.array(primary_inlet_bc[i])[0:index_stop][::subsampling_index]
        output_dict[name]['vessel_to_primary'][i]=np.array(primary_outlet_bc[i])[0:index_stop][::subsampling_index]
        
    for i in ['T_gas_primary_volume','x_alfa_primary_volume', 'P_steam_primary_volume', 'P_saturation_primary_volume', 'P_H2_primary_volume','P_primary_volume', 'm_steam_primary_volume', 'rho_liq_primary_volume', 'm_liq_primary_volume', 'T_sat_primary_volume', 'x_steam_primary_volume', 'T_liq_primary_volume']:
        output_dict[name]['VDO'][i]=np.array(VDO[i])[0:index_stop][::subsampling_index]
        output_dict[name]['UPP_V001'][i]=np.array(UPP_V001[i])[0:index_stop][::subsampling_index]
        
    next_time_step = f['dimensions/time_points'][1:index_stop]
    previous_time_step = f['dimensions/time_points'][0:index_stop-1]
    DT = next_time_step - previous_time_step
    DT = np.concatenate([next_time_step - previous_time_step, [DT[-1]]])[::subsampling_index]
    
    output_dict[name]['vessel_to_primary']['dt']=DT #used by AE_NODE
    output_dict[name]['vessel_to_primary']['time']=f['dimensions/time_points'][0:index_stop][::subsampling_index]#used by ONLY_DECODER

def extract_input_output_bc_variables(path, index_simulation:str, subsampling_index:int):
    output_dict = {}
    time_of_simulations = []
    name_simulation = str(index_simulation) + '.h5'
    with h5py.File(path+'/'+str(name_simulation), 'r') as f:
        vessel_rupture_time = f['other/global/vessel_rupture_time'][-1]
        if not np.isnan(vessel_rupture_time):
            index_stop = np.where(f['dimensions/time_points'][:] >= vessel_rupture_time)[0][0]
            index_stop = len(f['dimensions/time_points'][0:index_stop])
        else:
            index_stop = len(f['dimensions/time_points'][:])
        time_of_simulations.append(f['dimensions/time_points'][:][0:index_stop][::subsampling_index])   
        output_dict[index_simulation] = build_dictionary_of_variables()
        fill_dictionary_of_variables(output_dict, index_simulation, f, index_stop, subsampling_index)

    return output_dict, time_of_simulations

def extract_time_of_simulation(path, index_simulation:str, subsampling_index:int):
    name_simulation = str(index_simulation) + '.h5'
    with h5py.File(path+'/'+str(name_simulation), 'r') as f:
        vessel_rupture_time = f['other/global/vessel_rupture_time'][-1]
        if not np.isnan(vessel_rupture_time):
            index_stop = np.where(f['dimensions/time_points'][:] >= vessel_rupture_time)[0][0]
            index_stop = len(f['dimensions/time_points'][0:index_stop])
        else:
            index_stop = len(f['dimensions/time_points'][:])
        time_of_simulations = f['dimensions/time_points'][:][0:index_stop][::subsampling_index]

    return time_of_simulations

def add_dict_to_hdf5(h5_path, key, dictionary, path=''):
    """Add an entire dictionary to an existing HDF5 file under a specific key."""
    with h5py.File(h5_path, 'a') as h5file:
        key = str(key)
        full_path = f"{path}/{key}" if path else key
        
        # Check if key already exists and remove it
        if full_path in h5file:
            del h5file[full_path]
        
        # Create a group for this key
        if full_path not in h5file:
            h5file.create_group(full_path)
        
        # Use your existing dict_to_hdf5 function to populate it
        dict_to_hdf5(dictionary, h5file, full_path)

def dict_to_hdf5(dictionary, h5file, path=''):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_to_hdf5(value, h5file, f"{path}/{key}")
        else:
            # Handle PyTorch tensors
            if isinstance(value, tc.Tensor):
                value = value.cpu().numpy()
            # Handle other array-like objects
            elif hasattr(value, 'numpy'):
                value = value.numpy()
            
            h5file[f"{path}/{key}"] = value
            
def get_normalization_statistics_progressively(path_hdf5:str, type_of_normalization:str):
    exist = False
    total_time_steps = 0 
    maxima_or_mean = {}
    minima_or_std = {}
    with h5py.File(path_hdf5, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            simulation_maxima_or_mean, simulation_minima_or_std, lenght_simulation = get_normalization_statistics(f[key],type_of_normalization)
            total_time_steps += lenght_simulation
            if not exist:
                if type_of_normalization == 'min_max':
                    maxima_or_mean = simulation_maxima_or_mean
                    minima_or_std = simulation_minima_or_std
                elif type_of_normalization == 'mean_std':
                    for shape in simulation_maxima_or_mean:
                        maxima_or_mean[shape] = simulation_maxima_or_mean[shape] * lenght_simulation
                        minima_or_std[shape] = (simulation_minima_or_std[shape]**2 + simulation_maxima_or_mean[shape]**2) * lenght_simulation
                exist = True
            else:
                update_normalization_statistics(maxima_or_mean, minima_or_std, simulation_maxima_or_mean, simulation_minima_or_std,type_of_normalization , lenght_simulation)
    
    if type_of_normalization == 'mean_std':
        
        for shape in maxima_or_mean:
            maxima_or_mean[shape] = maxima_or_mean[shape] / total_time_steps
            minima_or_std[shape] = (minima_or_std[shape] / total_time_steps - maxima_or_mean[shape]**2)**0.5
            
    #check for constant values
    for shape in maxima_or_mean:
        for count, index in enumerate(maxima_or_mean[shape]):
            if type_of_normalization == 'mean_std':
                if minima_or_std[shape][count] == 0.0:
                    minima_or_std[shape][count] = maxima_or_mean[shape][count] if maxima_or_mean[shape][count] !=0.0 else 1.0
                    maxima_or_mean[shape][count] = 0.0
                    
            elif type_of_normalization == 'min_max':
                if maxima_or_mean[shape][count]-minima_or_std[shape][count] == 0.0:
                    if maxima_or_mean[shape][count] == 0.0:
                        maxima_or_mean[shape][count] = 1.0 
                    minima_or_std[shape][count] = 0.0
                    
    return maxima_or_mean, minima_or_std      
def update_normalization_statistics(maxima_or_mean:dict, minima_or_std:dict, simulation_maxima_or_mean:dict, simulation_minima_or_std:dict ,type_of_normalization:str, lenght_simulation:int ):
    if type_of_normalization == 'min_max':
        for shape in maxima_or_mean:
            maxima_or_mean[shape] = np.maximum(maxima_or_mean[shape],simulation_maxima_or_mean[shape])
            minima_or_std[shape] = np.maximum(minima_or_std[shape],simulation_minima_or_std[shape])
            
    elif type_of_normalization == 'mean_std':
        for shape in maxima_or_mean:
            maxima_or_mean[shape] += simulation_maxima_or_mean[shape] * lenght_simulation
            minima_or_std[shape] += (simulation_minima_or_std[shape]**2 + simulation_maxima_or_mean[shape]**2) * lenght_simulation
    
def get_normalization_statistics(simulation:dict, type_of_normalization:str):
    maxima_or_mean = {}
    minima_or_std = {}
    shapes = list(simulation.keys())
    if type_of_normalization == 'min_max':
        for shape in shapes:
            size = np.shape(simulation[shape])
            minimum = np.min(simulation[shape].astype(np.float64),axis = (0,) + tuple(np.arange(2,len(size))))
            maximum = np.max(simulation[shape].astype(np.float64),axis = (0,) + tuple(np.arange(2,len(size))))
            minima_or_std[shape] = minimum
            maxima_or_mean[shape] = maximum
        
        maxima_or_mean['boundary_conditions_and_time'][-2] = 1.0 #no normalization of dt
        minima_or_std['boundary_conditions_and_time'][-2] = 0.0
                    
    elif type_of_normalization == 'mean_std':
        for shape in shapes:
            size = np.shape(simulation[shape])
            mean = np.mean(simulation[shape].astype(np.float64), axis=(0,) + tuple(np.arange(2, len(size))))
            std = np.std(simulation[shape].astype(np.float64), axis=(0,) + tuple(np.arange(2, len(size))))
            minima_or_std[shape] = std
            maxima_or_mean[shape] = mean
            
        maxima_or_mean['boundary_conditions_and_time'][-2] = 0.0 #no normalization of dt
        minima_or_std['boundary_conditions_and_time'][-2] = 1.0
    

    else:
        raise TypeError("Type of normalization not known. It can either be min_max or mean_std")  
                    
    return maxima_or_mean, minima_or_std, np.shape(simulation[shapes[0]])[1]

def normalize_fields(field: np.array, maximum_or_mean: dict, minimum_or_std: dict, normalization: str, device):
    field = tc.tensor(field)
    size = field.size()
    field = field.to(device)
    if size[-1] == 32:
        
        maximum_or_mean = maximum_or_mean['boundary_conditions_and_time'].to(device)
        minimum_or_std = minimum_or_std['boundary_conditions_and_time'].to(device)
        maximum_or_mean = maximum_or_mean[None,:]
        minimum_or_std = minimum_or_std[None,:]
        
    elif size[-1] == 57:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_1'].to(device)
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_1'].to(device)
        maximum_or_mean = maximum_or_mean[None,:]
        minimum_or_std = minimum_or_std[None,:]

    elif size[-1] == 140:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_140'].to(device)
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_140'].to(device)
        maximum_or_mean = maximum_or_mean[None,:, None]
        minimum_or_std = minimum_or_std[None,:, None]

    elif size[-1] == 36:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_36'].to(device)
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_36'].to(device)
        maximum_or_mean = maximum_or_mean[None,:, None]
        minimum_or_std = minimum_or_std[None,:, None]

    elif size[-1] == 76:
        maximum_or_mean = maximum_or_mean['dictionary_of_input_variables_76'].to(device)
        minimum_or_std = minimum_or_std['dictionary_of_input_variables_76'].to(device)
        maximum_or_mean = maximum_or_mean[None,:, None]
        minimum_or_std = minimum_or_std[None,:, None]
        
    else:
        raise TypeError(f"Something is wrong with data, shape is {size[-1]}")
    
    if normalization == 'min_max':
        denom = maximum_or_mean - minimum_or_std
        field = field - minimum_or_std
        field /= denom
            
    elif normalization == 'mean_std':
        field = field - maximum_or_mean
        field /= (minimum_or_std)
        
    elif normalization == 'none':
        return field

    else:
        raise ValueError(f"Missing value") 
        
    return field

def squeeze_first_dimension(dictionary_per_trajectory:dict):
    for i in dictionary_per_trajectory:
        for k in dictionary_per_trajectory[i]:
            dictionary_per_trajectory[i][k] = dictionary_per_trajectory[i][k].squeeze(0)
    return dictionary_per_trajectory  

def build_total_sampling_percentages(percentages_sampling: list, subsampling: list, length: int ):
    total_training_sampling_percentages = []
    for count, i in enumerate(percentages_sampling):
        i_training_sampling_indeces = [subsampling[count] for x in range(round(length * i))]
        total_training_sampling_percentages += i_training_sampling_indeces
    return total_training_sampling_percentages


def fix_total_percentages(initial_percentages:list, difference:int):
    if difference > 0:
        for i in range(difference):
            initial_percentages.append(initial_percentages[-1])
        print(f"Added {difference} elements")
    elif difference < 0:
        initial_percentages = initial_percentages[:difference]
        print(f"Removed {np.abs(difference)} elements")
    elif difference == 0:
        print(f"No modifications")
    return initial_percentages