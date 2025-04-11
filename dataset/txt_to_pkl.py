import pandas as pd
import os
import argparse
import pickle
from typing import List, Tuple

def parse_yolo_annotation(txt_path: str) -> List[List[float]]:
    """
    Parse a YOLO annotation file and extract the bounding box coordinates.
    """
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Extract x_center and y_center
            x_center, y_center = map(float, parts[1:3])
            points.append([x_center, y_center])
    return points

def convert_txt_to_pkl(labels_dir: str, output_pkl: str):
    """
    Convert YOLO format txt annotations back into a pickle file.
    """
    data = []
    
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(labels_dir, txt_file)
            points = parse_yolo_annotation(txt_path)
            
            if not points:
                continue
            
            # Extract imagename
            img_name = txt_file.replace('.txt', '.jpg')

            #set img folder manually to test_v1
            img_folder = 'test_v1'

            data.append({'img_name': img_name, 'xy': points, 'img_folder': img_folder})
    
    # Convert to DataFrame and save as Pickle file
    df = pd.DataFrame(data)
    with open(output_pkl, 'wb') as f:
        pickle.dump(df, f, protocol=4)  # Use protocol 4 for compatibility with Python 3.7
    
    print(f"Conversion completed. Pickle file saved as {output_pkl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--labels-path', required=True, help='Path to YOLO txt labels directory')
    parser.add_argument('-o', '--output-path', required=True, help='Output pickle file path')
    
    args = parser.parse_args()
    convert_txt_to_pkl(args.labels_path, args.output_path)
