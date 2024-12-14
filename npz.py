import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

def select_images_from_dataset(dataset_dir, num_folders=200, images_per_folder=2, output_dir="selected_images"):
    """
    Selects a specified number of images from a specified number of folders in a dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    # Get all folders in the dataset directory
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    if len(folders) < num_folders:
        print(f"Only {len(folders)} folders found, but {num_folders} were requested.")
        num_folders = len(folders)

    selected_folders = random.sample(folders, num_folders)  # Randomly select folders
    total_selected = 0

    for folder in tqdm(selected_folders, desc="Selecting images"):
        folder_path = os.path.join(dataset_dir, folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) < images_per_folder:
            print(f"Folder {folder} has fewer than {images_per_folder} images.")
            continue

        selected_images = random.sample(image_files, images_per_folder)
        for img_file in selected_images:
            img_path = os.path.join(folder_path, img_file)
            output_path = os.path.join(output_dir, f"{folder}_{img_file}")
            try:
                Image.open(img_path).save(output_path)  # Save the selected image to the output directory
                total_selected += 1
            except Exception as e:
                print(f"Error processing file {img_file}: {e}")

    print(f"Selected {total_selected} images from {num_folders} folders.")
    return output_dir

def create_npz_from_sample_folder(sample_dir, output_dir, max_num=None):
    """
    Builds a single .npz file from all image files in a folder and saves it to a specified output directory.
    Resizes all images to 256x256 to ensure uniformity.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    samples = []
    files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files)  # Sort the files alphabetically
    if max_num:
        files = files[:max_num]

    for file_name in tqdm(files, desc="Building .npz file from samples"):
        file_path = os.path.join(sample_dir, file_name)
        try:
            sample_pil = Image.open(file_path).convert("RGB")  # Ensure 3 channels (RGB)
            sample_pil = sample_pil.resize((256, 256))  # Resize to 256x256
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

    if len(samples) == 0:
        print("No valid images found to create the .npz file.")
        return

    samples = np.stack(samples)  # Stack into a single numpy array
    npz_path = os.path.join(output_dir, "samples.npz")
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

# Paths and settings
dataset_directory = "/scratch/abhinav597project/RAT-Diffusion/dataset"  # Replace with the path to your dataset
output_images_directory = "/scratch/abhinav597project/npz_images"  # Temporary directory for selected images
output_npz_directory = "/scratch/abhinav597project"  # Directory to save the .npz file
num_folders_to_select = 200
images_per_folder = 6

# Step 1: Select images from the dataset
selected_images_dir = select_images_from_dataset(
    dataset_dir=dataset_directory,
    num_folders=num_folders_to_select,
    images_per_folder=images_per_folder,
    output_dir=output_images_directory
)

# Step 2: Create an .npz file from the selected images
create_npz_from_sample_folder(
    sample_dir=selected_images_dir,
    output_dir=output_npz_directory,
    max_num=None  # Process all images in the selected folder
)
