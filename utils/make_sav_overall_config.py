"""

Example use:
python utils/make_sav_overall_config.py /scratch/project_2005102/sophie/sav/sav_train_out/sav_train_2708
"""

import os
import pandas as pd
import argparse
from pathlib import Path

METADATA_FILENAME = 'metadata.csv' #'florence_base_captions.csv'

def find_metadata_files(root_dir):
    """Find all 'metadata.csv' files in subdirectories of root_dir."""
    metadata_files = []
    for subdir, _, files in os.walk(root_dir):
        if METADATA_FILENAME in files:
            metadata_files.append(os.path.join(subdir, METADATA_FILENAME))
    return metadata_files

def combine_metadata(root_dir, output_file=None):
    """
    Combine all 'metadata.csv' files from subdirectories into a single CSV.
    Adds the subdirectory name as its own column.
    
    Args:
        root_dir: Root directory containing subdirectories with metadata.csv files
        output_file: Path for the combined metadata file (default: root_dir/combined_metadata.csv)
    
    Returns:
        Path to the combined metadata file
    """
    if output_file is None:
        output_file = os.path.join(root_dir, f"root_{METADATA_FILENAME}")
    
    metadata_files = find_metadata_files(root_dir)
    
    if not metadata_files:
        print(f"No 'metadata.csv' files found in subdirectories of {root_dir}")
        return None
    
    dfs = []
    for file_path in metadata_files:
        try:
            df = pd.read_csv(file_path)
            
            subdirectory = os.path.basename(os.path.dirname(file_path))
            df['subdirectory'] = subdirectory
                
            dfs.append(df)
            print(f"Processed: {subdirectory}")

            ####### VALIDATION SPLIT #######
            # make 10% of the files in this dir a part of the validation split
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined metadata saved to: {output_file}")
        return output_file
    else:
        print("No valid metadata.csv files were processed")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine metadata.csv files from subdirectories")
    parser.add_argument("root_dir", help="Root directory containing subdirectories with metadata.csv files")
    parser.add_argument("--output", "-o", help="Output file path (default: root_dir/combined_metadata.csv)")
    
    args = parser.parse_args()
    
    combine_metadata(
        args.root_dir, 
        output_file=args.output
    )