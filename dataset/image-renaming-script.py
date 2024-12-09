import argparse
import os
import sys
from datetime import datetime
from PIL import Image, ExifTags

def get_image_creation_date(image_path):
    """
    Extract image creation date from various metadata sources.
    Falls back to file creation/modification time if no metadata is available.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Try to get EXIF data
        exif = img._getexif()
        
        if exif:
            # Dictionary of EXIF tags for creation/modification dates
            date_tags = {
                36867: 'DateTimeOriginal',  # Capture date
                36868: 'DateTimeDigitized', # Digitization date
                306:   'DateTime'           # Modification date
            }
            
            for tag_id, tag_name in date_tags.items():
                if tag_id in exif:
                    try:
                        # Parse date string
                        return datetime.strptime(exif[tag_id], '%Y:%m:%d %H:%M:%S')
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass
    
    # Fallback to file system timestamps
    try:
        # Prefer creation time, fall back to modification time
        stat = os.stat(image_path)
        creation_time = stat.st_ctime if hasattr(stat, 'st_ctime') else stat.st_mtime
        return datetime.fromtimestamp(creation_time)
    except Exception:
        # Last resort: use current time
        return datetime.now()

def rename_images_in_dataset(base_path='images'):
    """
    Rename images across all subfolders in the base path.
    Images are renamed with format: dd_mm_yyyy_IMG_id.ext
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
    
    # First pass: collect all image paths with their creation dates
    image_info = []
    for root, _, files in os.walk(base_path):
        for filename in files:
            # Check if file is an image
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, filename)
                try:
                    creation_date = get_image_creation_date(full_path)
                    image_info.append((full_path, creation_date))
                except Exception as e:
                    print(f"Could not process {full_path}: {e}")
    
    # Sort images by creation date
    image_info.sort(key=lambda x: x[1])
    
    # Second pass: rename images
    for global_id, (image_path, creation_date) in enumerate(image_info, start=1):
        # Extract file extension
        file_ext = os.path.splitext(image_path)[1]
        
        # Create new filename
        #new_filename = f"{creation_date.day:02d}_{creation_date.month:02d}_{creation_date.year}_IMG_{global_id}{file_ext}"
        new_filename= f"no_crop_IMG_{global_id}{file_ext}"
        # Get directory of the current image
        directory = os.path.dirname(image_path)
        
        # Full path for new filename
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(image_path, new_path)
        print(f"Renamed: {image_path} -> {new_path}")
    
    print(f"Processed {len(image_info)} images.")

# Usage
def main(folder_path):
    # Specify the base path where your image folders are located
    base_path = folder_path
    
    # Confirm before running
    print(f"This script will rename all images in {base_path} and its subfolders.")
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower()
    
    if confirm == 'yes':
        rename_images_in_dataset(base_path)
        print("Image renaming completed.")
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rename images in a dataset')
    parser.add_argument('-f', '--folder-path', default='images', help='Path to the folder containing images')
    args = parser.parse_args()
    
    main(args.folder_path)