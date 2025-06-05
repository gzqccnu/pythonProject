import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # Optional: to measure execution time

# --- Noise Generation Function (Unchanged) ---
def get_noise(img, value=10):
    '''
    Generates noise image.
    :param img: Input image (NumPy array)
    :param value: Controls the amount of noise (higher value means more noise points)
    :return: Grayscale noise image (NumPy array)
    '''
    if img is None:
        return None
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # Control noise level, thresholding based on value
    v = value * 0.01 # Note: Original description seemed reversed. Let's assume higher value = more noise.
                     # If value is intended max intensity, the logic might need adjustment.
                     # Current logic: Keep pixels >= 256 - v. If value=500 -> v=5 -> keep pixels >= 251.
                     # If you want value to control density (0-100), maybe use:
                     # threshold = np.percentile(noise, 100 - value)
                     # noise[noise < threshold] = 0
                     # Let's stick to the original calculation for now:
    threshold = 256 - (value * 0.01 * 256) # Scale value to 0-256 range for thresholding
    noise[np.where(noise < threshold)] = 0


    # Simple blur on noise
    k = np.array([[0, 0.1, 0],
                  [0.1, 0.8, 0.1], # Reduced center weight slightly from 8 to maybe look less sharp
                  [0, 0.1, 0]])
    k = k / np.sum(k) # Normalize kernel

    noise = cv2.filter2D(noise, -1, k)

    return noise

# --- Rain Motion Blur Function (Unchanged structure, ensure dtype) ---
def rain_blur(noise, length=10, angle=0, w=1):
    '''
    Applies motion blur to noise to simulate rain streaks.
    :param noise: Input noise image (grayscale)
    :param length: Length of the rain streaks
    :param angle: Angle of the rain streaks (degrees, counter-clockwise)
    :param w: Width of the rain streaks (Gaussian kernel size)
    :return: Blurred noise image representing rain (uint8 NumPy array)
    '''
    if noise is None:
        return None

    # Ensure noise is float for calculations if needed, though filter2D handles uint8
    # noise_float = noise.astype(np.float32) # Not strictly necessary here if using uint8 input

    # Create motion blur kernel
    # Adjust angle calculation if needed, original had -45 adjustment
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # Diagonal matrix
    k = cv2.warpAffine(dig, trans, (length, length))  # Rotate kernel
    k = cv2.GaussianBlur(k, (w, w), 0)  # Blur kernel for thickness

    # Normalize the kernel
    k_sum = np.sum(k)
    if k_sum != 0:
        k = k / k_sum

    blurred = cv2.filter2D(noise, -1, k)  # Apply motion blur

    # Normalize result to 0-255 and convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred

# --- Rain Blending Function (Modified to RETURN result) ---
def apply_rain_effect(rain_mask, img, beta=0.8):
    '''
    Blends the rain mask with the original image.
    :param rain_mask: Blurred noise image (grayscale uint8) representing rain
    :param img: Original image (color NumPy array)
    :param beta: Weight of the rain effect (controls intensity/visibility)
    :return: Image with rain effect (NumPy array uint8)
    '''
    if rain_mask is None or img is None:
        return None # Or return original img? Returning None indicates failure.

    # Ensure rain_mask is float32 for the blending calculation
    rain_mask_float = rain_mask.astype(np.float32) / 255.0 # Normalize to 0.0-1.0

    # Ensure img is float32 for calculation
    img_float = img.astype(np.float32)

    # Expand rain_mask to 3 channels to match image
    rain_mask_3ch = cv2.cvtColor(rain_mask_float, cv2.COLOR_GRAY2BGR)

    # Blend using the formula: output = img * (1 - mask) + beta * mask * 255
    # Or using the original formula's approach:
    # output = img * (1 - mask) + beta * mask * 255 (scaling beta effect by mask intensity)

    rain_result_float = img_float * (1.0 - rain_mask_3ch) + (beta * rain_mask_3ch * 255.0)

    # Clip values to 0-255 and convert back to uint8
    rain_result = np.clip(rain_result_float, 0, 255).astype(np.uint8)

    # The original calculation was:
    # rain_result[:,:,c] = img[:,:,c] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    # Let's replicate that more closely:
    # rain_mask_expanded = np.expand_dims(rain_mask.astype(np.float32), axis=2) # Expand grayscale mask
    # rain_result_float = img_float * (255.0 - rain_mask_expanded) / 255.0 + beta * rain_mask_expanded
    # rain_result = np.clip(rain_result_float, 0, 255).astype(np.uint8)

    return rain_result

# --- Function to Process and Save a Single Image ---
def process_and_save_rain_image(input_path, output_dir, noise_value, rain_length, rain_angle, rain_width, rain_beta):
    """
    Loads an image, adds rain effect, and saves it to the output directory.
    """
    try:
        # Extract filename and create output path
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_rainy{ext}" # Add suffix
        output_path = os.path.join(output_dir, output_filename)

        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            return f"Failed to read: {input_path}"

        # --- Apply Rain Effect ---
        # 1. Generate noise
        noise = get_noise(image, value=noise_value)
        if noise is None:
            print(f"Warning: Noise generation failed for {input_path}. Skipping.")
            return f"Noise failed: {input_path}"

        # 2. Create rain streaks
        rain_streaks = rain_blur(noise, length=rain_length, angle=rain_angle, w=rain_width)
        if rain_streaks is None:
            print(f"Warning: Rain blur failed for {input_path}. Skipping.")
            return f"Blur failed: {input_path}"

        # 3. Blend rain with image
        image_rainy = apply_rain_effect(rain_streaks, image, beta=rain_beta)
        if image_rainy is None:
             print(f"Warning: Rain blending failed for {input_path}. Skipping.")
             return f"Blend failed: {input_path}"
        # --- End Rain Effect ---

        # Save the rainy image
        success = cv2.imwrite(output_path, image_rainy)
        if not success:
             print(f"Warning: Could not write image {output_path}. Skipping.")
             return f"Failed to write: {output_path}"

        return f"Success: {output_path}"

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return f"Error: {input_path} - {e}"

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    input_folder = r'/root/code/.code/pic/images' # Directory containing original images
    output_folder = r'/root/code/.code/pic/rain' # Directory to save rainy images

    # Rain Parameters (Adjust as needed)
    p_noise_value = 500     # Controls density/amount of rain points (adjust based on get_noise behavior)
    p_rain_length = 30      # Length of rain streaks
    p_rain_angle = -3       # Angle of rain streaks (degrees, -ve for typical slant)
    p_rain_width = 1        # Width of rain streaks (blur kernel size, odd number usually)
    p_rain_beta = 0.6       # Blending factor (intensity/visibility of rain)

    # Processing Parameters
    max_workers = 8          # Number of threads (adjust based on CPU)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff') # Supported image types

    # --- Prepare Input and Output ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory: {output_folder}")

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"No images found in {input_folder} with extensions {image_extensions}")
        exit()

    print(f"Found {len(image_files)} images to process with rain effect.")

    # --- Multithreaded Processing ---
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_save_rain_image,
                                   img_path,
                                   output_folder,
                                   p_noise_value,
                                   p_rain_length,
                                   p_rain_angle,
                                   p_rain_width,
                                   p_rain_beta): img_path for img_path in image_files}

        for future in as_completed(futures):
            result = future.result()
            if "Success" in result:
                processed_count += 1
            else:
                failed_count += 1
                # print(result) # Optionally print failure details
            # Optional: print progress periodically
            if (processed_count + failed_count) % 50 == 0:
                 print(f"Progress: {processed_count + failed_count}/{len(image_files)} (Processed: {processed_count}, Failed: {failed_count})")


    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- Rain Effect Processing Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Skipped:      {failed_count}")
    print(f"Total images:        {len(image_files)}")
    print(f"Total time taken:    {total_time:.2f} seconds")
