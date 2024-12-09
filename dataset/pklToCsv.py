import pandas as pd
import os
import pickle
import json
import argparse
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy arrays and other NumPy types.
    Converts NumPy arrays to lists and handles NumPy numeric types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def get_filename_without_extension(file_path):
    """
    Extract the filename without its extension using os.path.
    
    Args:
        file_path (str): Full file path
    
    Returns:
        str: Filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def convert_pkl_to_csv(pkl_file_path):
    """
    Convert a pickle file to CSV or other formats, handling various data types.
    
    Args:
        pkl_file_path (str): Path to the pickle file
    """
    # Normalize the file path to work across different platforms
    pkl_file_path = os.path.normpath(pkl_file_path)
    
    # Get the folder path and file name
    folder_path = os.path.dirname(pkl_file_path)
    file_name = os.path.basename(pkl_file_path)
    
    try:
        # Try loading the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Determine how to save based on data type
        file_name_without_ext = get_filename_without_extension(pkl_file_path)
        output_base_path = os.path.join(folder_path, file_name_without_ext)
        
        if isinstance(data, pd.DataFrame):
            # If it's a DataFrame, save directly to CSV
            output_csv_path = output_base_path + '.csv'
            print(f"Saving DataFrame to {output_csv_path}")
            data.to_csv(output_csv_path, index=True)
        
        elif isinstance(data, (dict, np.ndarray)):
            # If it's a dictionary or NumPy array, handle carefully
            
            # Option 1: Save as JSON with custom encoder
            output_json_path = output_base_path + '.json'
            print(f"Saving data to {output_json_path}")
            with open(output_json_path, 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
            
            # Option 2: If dictionary contains DataFrames, save each as CSV
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        # Sanitize key to make it file-system friendly
                        safe_key = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(key))
                        output_df_path = f"{output_base_path}_{safe_key}.csv"
                        print(f"Saving DataFrame from key '{key}' to {output_df_path}")
                        value.to_csv(output_df_path, index=True)
                    elif isinstance(value, np.ndarray):
                        # Save NumPy arrays as CSV 
                        safe_key = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(key))
                        output_array_path = f"{output_base_path}_{safe_key}.csv"
                        print(f"Saving NumPy array from key '{key}' to {output_array_path}")
                        
                        # Convert to DataFrame for easy CSV saving if 2D
                        if value.ndim == 2:
                            pd.DataFrame(value).to_csv(output_array_path, index=False)
                        else:
                            # For higher-dimensional arrays, save as list in JSON
                            with open(output_array_path, 'w') as f:
                                json.dump(value.tolist(), f, indent=4)
        
        else:
            # For other types, save as pickle
            output_pkl_path = output_base_path + '.pkl'
            print(f"Unsupported data type: {type(data)}. Saving as pickle to {output_pkl_path}")
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(data, f)
    
    except Exception as e:
        print(f"Error processing {pkl_file_path}: {e}")
        raise

def main(pkl_file_path):
    """
    Main function to convert pickle file.
    
    Args:
        pkl_file_path (str): Path to the pickle file
    """
    convert_pkl_to_csv(pkl_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pickle files to various formats')
    parser.add_argument('-pkl', '--pkl-path', default='labels.pkl', 
                        help='Path to the pickle file to convert')
    args = parser.parse_args()
    
    main(args.pkl_path)