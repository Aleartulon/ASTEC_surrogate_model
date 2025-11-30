import h5py
import os

def rename_to_simulation_name(filepath):
    with h5py.File(filepath, 'r') as f:
        archive_name = f.attrs['archive_name'].decode('utf-8')
    
    directory = os.path.dirname(filepath)
    new_name = f"{archive_name}.h5"
    new_path = os.path.join(directory, new_name)
    
    os.rename(filepath, new_path)
    print(f"Renamed: {os.path.basename(filepath)} → {new_name}")

def rename_all_in_directory(directory):
    # Get all .h5 files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5') and filename.startswith('dataset_'):
            filepath = os.path.join(directory, filename)
            try:
                rename_to_simulation_name(filepath)
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Usage
directory = '../../../../../../../scratch/aalelonghi/ROM_datasets_ale/ASTEC/original_hdf5/'  # Current directory, or specify your path like '/path/to/your/files'
rename_all_in_directory(directory)