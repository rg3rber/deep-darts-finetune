import os

def check_images_and_labels(image_folder, label_folder):
    # Define common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # Track missing labels and orphaned labels
    missing_labels = []
    orphaned_labels = []

    # Create a set of image base names (without extensions)
    image_base_names = {
        os.path.splitext(image_file)[0]
        for image_file in os.listdir(image_folder)
        if image_file.lower().endswith(image_extensions)
    }

    # Check for missing labels (images without corresponding label files)
    for image_base in image_base_names:
        label_file = f"{image_base}.txt"
        label_path = os.path.join(label_folder, label_file)
        if not os.path.exists(label_path):
            missing_labels.append(f"{image_base}.jpg (or other extension)")

    # Check for orphaned labels (labels without corresponding images)
    for label_file in os.listdir(label_folder):
        if label_file.lower().endswith('.txt'):
            label_base = os.path.splitext(label_file)[0]
            if label_base not in image_base_names:
                orphaned_labels.append(label_file)

    # Print results
    if missing_labels:
        print("The following images are missing corresponding label files:")
        for image_file in missing_labels:
            print(f"- {image_file}")
    else:
        print("All images have corresponding label files.")

    if orphaned_labels:
        print("\nThe following label files do not have corresponding images:")
        for label_file in orphaned_labels:
            print(f"- {label_file}")
    else:
        print("\nAll label files have corresponding images.")

if __name__ == "__main__":
    # Ask the user for the folder paths
    image_folder = input("Enter the path to the folder containing images: ").strip('"')
    label_folder = os.path.join(os.path.dirname(image_folder), "labels")  # Assume "labels" is one level above

    if os.path.isdir(image_folder) and os.path.isdir(label_folder):
        check_images_and_labels(image_folder, label_folder)
    else:
        print("Invalid folder paths. Please ensure the image folder exists and the labels folder is one level above.")
