import os
import sys
import torch as tc
import yaml
import shutil
import importlib
from src.common_functions import load_config

#set type of tensors
tc.set_default_dtype(tc.float32)


def main():
    
    print(f"PID process: {os.getpid()}")
    if tc.cuda.is_available():
        tc.backends.cuda.matmul.allow_tf32 = True
        tc.backends.cudnn.allow_tf32 = True

    config_training = load_config('configs/config_training.yaml')
    model_information = load_config('configs/configs_models/' + 'config_'+config_training['model'] + '.yaml')
    
    # Override where_to_save_data with local node SSD if available
    local_data = os.environ.get('LOCAL_DATA_PATH', None)
    if local_data:
        print(f"\n{'='*60}")
        print(f"Overriding where_to_save_data with local SSD: {local_data}")
        print(f"Previously: {config_training['where_to_save_data']}")
        config_training['where_to_save_data'] = local_data + '/'
        print(f"Now: {config_training['where_to_save_data']}")
        print(f"{'='*60}\n")

    #define directories and get data

    config_training['PATH'] = config_training['physics_model']+ config_training['model'] +'/Models/' + config_training['description']

    #encode information in txt files and delete path where losses are saved to avoid having data from different runs
    
    if os.path.exists(config_training['PATH'] +'/runs'):
        shutil.rmtree(config_training['PATH']+'/runs') 

    # Copy the entire src directory
    shutil.copytree('src/models/AE_NODE/training', os.path.join(config_training['PATH'], 'scripts/src'), dirs_exist_ok=True)

    # Copy config
    shutil.copytree('configs', os.path.join(config_training['PATH'], 'scripts/configs'), dirs_exist_ok=True)

    # go to gpu if possible
    device = tc.device(config_training['device']) if tc.cuda.is_available() else tc.device("cpu")
    if device.type == 'cuda':
        tc.cuda.set_device(device.index or 0)  # Use index if specified, otherwise 0
    config_training['device'] = str(device)
    print(f'Selected device: {device}')

    #define the model
    module_name = 'src.models.'+str(config_training['model'])+'.training.'+str(config_training['model'])+'_model'
    class_name = config_training["model"]
    module = importlib.import_module(module_name)
    ClassRef = getattr(module, class_name)
    model = ClassRef(config_training, model_information)
    
    #print information

    print('---------- INITIAL INFORMATION ----------')
    for key, value in config_training.items():
        print(key, ' : ', value)
    print(" ")
    print('---------- MODEL INFORMATION ----------')
    for key, value in model_information.items():
        print(key, ' : ', value)
    print(" ")

    model.start_training()
    
if __name__ == '__main__':
    main()
