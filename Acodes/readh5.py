import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
# import matplotlib.font_manager as fm # No need for font_manager if using English and saving directly

# --- Matplotlib Configuration for Non-GUI Backend ---
# This is crucial for running on a server without a display.
# 'Agg' is a non-interactive backend that can write to files.
plt.rcParams['agg.path.chunksize'] = 10000 # Optional: Helps with large plots/complex paths
plt.switch_backend('Agg') 

# --- Matplotlib Standard Settings ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = True 
# ----------------------------------------------------

def visualize_megadepth_depth_and_save(h5_file_path, output_image_path):
    """
    Reads an H5 depth map file from the MegaDepth dataset, visualizes it,
    and saves the visualization as an image file.

    Args:
        h5_file_path (str): The full path to the H5 input file.
        output_image_path (str): The full path where the output image will be saved (e.g., 'depth_map.png').
    """
    if not os.path.exists(h5_file_path):
        print(f"Error: Input H5 file not found at path {h5_file_path}")
        return

    print(f"Processing H5 file: {h5_file_path}")
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            depth_key = 'depth'
            if depth_key in f:
                depth_map = f[depth_key][:]
            else:
                print(f"Error: Key '{depth_key}' not found in the H5 file.")
                print("Available keys in the file are:", list(f.keys()))
                return

        depth_map[depth_map == 0] = np.nan # Replace 0s with NaN for better visualization
        
        plt.figure(figsize=(12, 8)) # Create a figure
        
        v_min = np.nanmin(depth_map)
        v_max = np.nanmax(depth_map)
        
        plt.imshow(depth_map, cmap='jet', vmin=v_min, vmax=v_max)
        
        # Add color bar
        plt.colorbar(label='Depth (meters)')
        
        # Set title and labels
        plt.title(f"MegaDepth Depth Map Visualization\nInput File: {os.path.basename(h5_file_path)}")
        plt.xlabel('X Pixel Coordinate')
        plt.ylabel('Y Pixel Coordinate')
        plt.axis('off') # Hide axes for cleaner image
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # --- Save the figure to a file ---
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight') # dpi for resolution, bbox_inches for tight crop
        plt.close() # Close the figure to free up memory

        print(f"Visualization successfully saved to: {output_image_path}")

    except Exception as e:
        print(f"An error occurred while processing and saving the file: {e}")

# --- Usage Example (Specify Paths Here) ---
if __name__ == "__main__":
    
    # ---------------------------------------------------------------------------------------
    # !!! IMPORTANT: CONFIGURE YOUR INPUT AND OUTPUT PATHS BELOW !!!
    # ---------------------------------------------------------------------------------------
    
    # 1. Input H5 Depth Map File Path
    #    Replace this with the full absolute path to your .h5 depth map file.
    input_h5_file = "/home/data1/szq/Megadepth/benchemarkdata/Megadepth-aerial/0050-/dense0/depths/9433304_5e2830c65e_o.h5" 

    # 2. Output Image File Path
    #    Specify where you want to save the output image.
    #    You can use .png, .jpg, .pdf etc. PNG is recommended for quality.
    #    Example: Save in the same directory as the script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_image_filename = os.path.basename(input_h5_file).replace('.h5', '_depth_vis.png')
    output_image_file = os.path.join(script_dir, output_image_filename)

    # Example: Save in a specific output directory
    # output_dir = "/path/to/your/output_images/"
    # os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    # output_image_file = os.path.join(output_dir, output_image_filename)
    
    print(f"Input H5 file: {input_h5_file}")
    print(f"Output image will be saved to: {output_image_file}")

    # Run the visualization and save function
    visualize_megadepth_depth_and_save(input_h5_file, output_image_file)