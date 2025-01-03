import pandas as pd
from pathlib import Path
import os
import pickle


def getFileNameWithoutExtension(file_name):
    return os.path.splitext(file_name)[0]

def getFileName(file_path):
    return Path(file_path).name

def convert_pkl_to_csv(pkl_file_path):
    # Get the folder path and file name
    folder_path = Path(pkl_file_path).parent
    file_name = getFileName(pkl_file_path)
    # Load the data from the pickle file
    file_path = str(Path(folder_path, file_name))
    data = pd.read_pickle(file_path)
    # Save the data to a csv file
    file_name = getFileNameWithoutExtension(file_name)
    file_path = str(Path(folder_path, file_name))
    data.to_csv(f"{file_path}.csv")

def main():
    pkl_file_path = './labels.pkl'
    convert_pkl_to_csv(pkl_file_path)

if __name__ == '__main__':
    main()







