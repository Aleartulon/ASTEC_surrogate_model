import h5py
import os
import re

def rename_to_simulation_name(filepath):
    with h5py.File(filepath, 'r') as f:
        archive_name = f.attrs['archive_name'].decode('utf-8')
    
    # Look for digits at the end of the archive_name (1-3 digits)
    match = re.search(r'^(.+?)(\d{1,3})$', archive_name)
    if match:
        base_name = match.group(1)
        number = match.group(2)
        # Pad the number to 3 digits
        padded_number = number.zfill(3)
        archive_name = f"{base_name}{padded_number}"
    
    directory = os.path.dirname(filepath)
    new_name = f"{archive_name}.h5"
    new_path = os.path.join(directory, new_name)
    os.rename(filepath, new_path)
    print(f"Renamed: {os.path.basename(filepath)} → {new_name}")

def rename_all_in_directory(directory):
    # Get all .h5 files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            filepath = os.path.join(directory, filename)
            try:
                rename_to_simulation_name(filepath)
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Usage
directory = '../../../../../../../scratch/aalelonghi/ROM_datasets_ale/ASTEC/original_hdf5/'
rename_all_in_directory(directory)