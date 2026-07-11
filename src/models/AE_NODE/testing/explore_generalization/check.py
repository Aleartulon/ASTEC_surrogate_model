import numpy as np
import h5py

def main():
    extract_op = True
    #first extract all the operator actions of the available data if extract_op is true
    
    if extract_op:
        where_to_look = '../../../../../../tudelft.net/staff-umbrella/eldar/ASTEC_2/ordered_hdf5/'
        dictionary_of_op = associate_op_to_number(where_to_look)
    else:
        #obtain dictionary_of_operator_actions via loading
        directory_dictionary = 'dictionary_op.py'
        dictionary_of_op = np.load(directory_dictionary)
        
    return 0

def associate_op_to_number(path_to_directory):
    #look inside path_to_directory and look for the files ending with .h5. store their path in list_of_paths
    list_of_paths = []
    dictionary_of_op = {}
    for path_trajectory in list_of_paths:
        number_trajectory = 0
        operator_actions_list = extract_op_from_file(path_trajectory)
        dictionary_of_op[tuple(operator_actions_list)] = number_trajectory
    return dictionary_of_op

def extract_op_from_file(path_to_dataset:str):
    operator_actions_list = []
    with h5py.File(path_to_dataset, 'r') as f:
        operator_names = ['t_fbseb', 't1_srv', 'opensrv', 't2_srv', 'tendssg2', 'tpesp','tpessg', 'tcss', 'p_u5', 'tsg2tr']
        for op in operator_names:
            operator_actions_list.append((f['other/private/'+ op][0])/ 3600.0)
            if np.isnan(f['other/private/'+ op][0]).any() or not np.isfinite(f['other/private/'+ op][0]).any():
                raise TypeError(f"Operator action {op} in simulation {path_to_dataset} is NaN")
    return operator_actions_list