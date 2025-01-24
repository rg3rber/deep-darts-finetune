import os
import shutil

def move_images_to_root(root_folder):
    # Define common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):  # Check if the file is an image
                source_path = os.path.join(dirpath, filename)
                destination_path = os.path.join(root_folder, filename)

                # Avoid overwriting files with the same name
                if os.path.exists(destination_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(destination_path):
                        destination_path = os.path.join(root_folder, f"{base}_{counter}{ext}")
                        counter += 1

                # Move the file to the root folder
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")

    # Remove empty directories
    for dirpath, dirnames, _ in os.walk(root_folder, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):  # Check if directory is empty
                os.rmdir(dir_to_check)
                print(f"Removed empty folder: {dir_to_check}")

if __name__ == "__main__":
    root_folder = input("Enter the path to the root folder: ").strip('"')
    if os.path.isdir(root_folder):
        move_images_to_root(root_folder)
        print("All images have been moved to the root folder.")
    else:
        print("Invalid folder path. Please check and try again.")
