#!/usr/bin/env python3
"""
Script to rename all .h5 files in a directory to numbered format (1.h5, 2.h5, etc.)
in the order they appear in the directory, and create a log file mapping new names to original names.
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def rename_files_in_directory(directory_path, file_extension='.h5'):
    """
    Rename all files with the specified extension in directory order.
    
    Args:
        directory_path: Path to the directory containing files
        file_extension: Extension of files to rename (default: '.h5')
    """
    # Convert to Path object and resolve to absolute path
    directory = Path(directory_path).resolve()
    
    # Check if directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        return
    
    # Get all files with the specified extension
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix == file_extension]
    
    if not files:
        print(f"No {file_extension} files found in '{directory_path}'.")
        return
    
    # Sort files by name (as they appear in directory listing)
    files.sort(key=lambda x: x.name)
    
    print(f"Found {len(files)} {file_extension} files to rename.")
    
    # Create a temporary mapping to avoid naming conflicts
    rename_mapping = []
    temp_renames = []
    
    # First pass: rename to temporary names to avoid conflicts
    for idx, file_path in enumerate(files, start=1):
        original_name = file_path.name
        temp_name = f"_temp_{idx}{file_extension}"
        temp_path = file_path.parent / temp_name
        
        rename_mapping.append((idx, original_name))
        temp_renames.append((file_path, temp_path))
    
    # Perform temporary renames
    print("\nStep 1: Renaming to temporary names...")
    for original_path, temp_path in temp_renames:
        try:
            if not original_path.exists():
                print(f"Warning: {original_path} does not exist, skipping...")
                continue
            original_path.rename(temp_path)
            print(f"  Renamed: {original_path.name} -> {temp_path.name}")
        except Exception as e:
            print(f"Error renaming {original_path} to {temp_path}: {e}")
            raise
    
    # Second pass: rename from temporary names to final names
    print("\nStep 2: Renaming to final numbered names...")
    for idx, (_, temp_path) in enumerate(temp_renames, start=1):
        final_name = f"{idx}{file_extension}"
        final_path = directory / final_name
        try:
            if not temp_path.exists():
                print(f"Warning: {temp_path} does not exist!")
                continue
            temp_path.rename(final_path)
            print(f"  Renamed: {temp_path.name} -> {final_path.name}")
        except Exception as e:
            print(f"Error renaming {temp_path} to {final_path}: {e}")
            raise
    
    # Create log file
    log_filename = directory / "rename_log.txt"
    with open(log_filename, 'w') as log_file:
        log_file.write(f"File Renaming Log\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Directory: {directory.absolute()}\n")
        log_file.write(f"Total files renamed: {len(rename_mapping)}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for new_number, original_name in rename_mapping:
            log_file.write(f"{new_number} corresponds to: {original_name}\n")
    
    print(f"\nRenaming complete!")
    print(f"Log file created: {log_filename}")
    print("\nMapping:")
    for new_number, original_name in rename_mapping:
        print(f"  {new_number}{file_extension} <- {original_name}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python rename_files.py <directory_path> [file_extension]")
        print("Example: python rename_files.py /path/to/directory")
        print("Example: python rename_files.py /path/to/directory .txt")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    file_extension = sys.argv[2] if len(sys.argv) > 2 else '.h5'
    
    # Make sure extension starts with a dot
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    
    rename_files_in_directory(directory_path, file_extension)


if __name__ == "__main__":
    main()