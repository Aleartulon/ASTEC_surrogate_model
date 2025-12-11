import h5py
import os
import re

def get_simulation_name(filepath):
    """Extract and format the simulation name from the h5 file."""
    with h5py.File(filepath, 'r') as f:
        archive_name = f.attrs['archive_name'].decode('utf-8')
    
    match = re.search(r'^(.+?)(\d{1,3})$', archive_name)
    if match:
        base_name = match.group(1)
        number = match.group(2)
        padded_number = number.zfill(3)
        archive_name = f"{base_name}{padded_number}"
    
    new_name = f"{archive_name}.h5"
    return new_name

def write_filenames_to_txt(directory, output_file='file_mapping.txt'):
    """Write current filename and proposed new filename to a text file."""
    
    # Get absolute path
    output_file = os.path.abspath(output_file)
    print(f"Writing output to: {output_file}")
    
    with open(output_file, 'w') as txtfile:
        txtfile.write("Original Filename -> Proposed New Filename\n")
        txtfile.write("=" * 60 + "\n\n")
        
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.h5'):
                filepath = os.path.join(directory, filename)
                try:
                    new_name = get_simulation_name(filepath)
                    txtfile.write(f"{filename} -> {new_name}\n")
                    print(f"Logged: {filename} -> {new_name}")
                    count += 1
                except Exception as e:
                    error_msg = f"Error processing {filename}: {e}"
                    txtfile.write(f"{error_msg}\n")
                    print(error_msg)
    
    print(f"\nProcessed {count} files")
    print(f"File mapping written to: {output_file}")

# Main execution
directory = '../../../../../../tudelft.net/staff-umbrella/eldar/ASTEC/original_hdf5/'
write_filenames_to_txt(directory)
