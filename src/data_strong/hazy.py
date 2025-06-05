import numpy as np
import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # Optional: to measure execution time

# --- The Haze Adding Function (Unchanged) ---
def add_hazy(image, beta=0.05, brightness=0.5):
    '''
    Adds haze effect to an image.
    :param image: Input image (NumPy array)
    :param beta: Haze strength
    :param brightness: Haze brightness
    :return: Hazy image (NumPy array)
    '''
    if image is None:
        return None # Handle cases where image loading failed

    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))
    center = (row // 2, col // 2)
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return hazy_img

# --- Function to Process a Single Image ---
def process_and_save_image(input_path, output_dir, beta, brightness):
    """
    Loads an image, adds haze, and saves it to the output directory.
    """
    try:
        # Extract filename and create output path
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_hazy{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            return f"Failed to read: {input_path}"

        # Add haze
        image_fog = add_hazy(image, beta=beta, brightness=brightness)
        if image_fog is None:
             print(f"Warning: Haze processing failed for {input_path}. Skipping.")
             return f"Processing failed: {input_path}"

        # Save the hazy image
        success = cv2.imwrite(output_path, image_fog)
        if not success:
             print(f"Warning: Could not write image {output_path}. Skipping.")
             return f"Failed to write: {output_path}"

        # print(f"Processed and saved: {output_path}") # Optional: print progress for each file
        return f"Success: {output_path}"

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return f"Error: {input_path} - {e}"

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    input_folder = r'/root/code/.code/pic/images' # Directory containing original images
    output_folder = r'/root/code/.code/pic/hazy' # Directory to save hazy images
    haze_beta = 0.1          # Haze strength
    haze_brightness = 0.8    # Haze brightness
    max_workers = 8          # Number of threads to use (adjust based on your CPU cores)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff') # Supported image types

    # --- Prepare Input and Output ---
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory: {output_folder}")

    # Find all image files in the input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"No images found in {input_folder} with extensions {image_extensions}")
        exit()

    print(f"Found {len(image_files)} images to process.")

    # --- Multithreaded Processing ---
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    # Use ThreadPoolExecutor for managing threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks: process_and_save_image for each file
        futures = {executor.submit(process_and_save_image, img_path, output_folder, haze_beta, haze_brightness): img_path for img_path in image_files}

        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            if "Success" in result:
                processed_count += 1
            else:
                failed_count += 1
            # Optional: print progress periodically or use tqdm library for a progress bar
            if (processed_count + failed_count) % 50 == 0: # Print status every 50 images
                 print(f"Progress: {processed_count + failed_count}/{len(image_files)} (Processed: {processed_count}, Failed: {failed_count})")


    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Skipped:      {failed_count}")
    print(f"Total images:        {len(image_files)}")
    print(f"Total time taken:    {total_time:.2f} seconds")
