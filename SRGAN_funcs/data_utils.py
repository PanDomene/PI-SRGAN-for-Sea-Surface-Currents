import os
import numpy as np
import tensorflow as tf

#!!!!!!!!!!!!!!!!!!!! GENERALIZAR !!!!!!!!!!!!!!!!!!!!!!!!
def load_and_combine_channels(data_folder, third_channel=None):
    """
    Loads vx and vy data from the 'vx', 'vy' and 'third_channel' subfolders within data_folder,
     and combines them into a 3-channel image without performing any normalization.
    
    It assumes that vx files are named like "vx_{i}.npy", vy files like "vy_{i}.npy", etc.

    If third_channel is None, the third channel is filled with zeros.
    
    Args:
        - data_folder (str): Path to the folder containing the 'vx', 'vy' and 'third_channel' subfolders.
        - third_channel (str): The name of the folder containing the data for the 3d channel.
    Returns:
        List of combined images as NumPy arrays.
    """
    vx_dir = os.path.join(data_folder, "vx")
    vy_dir = os.path.join(data_folder, "vy")
    if third_channel:
        x_dir = os.path.join(data_folder, third_channel)
    # List and sort vx files (assumed to be in the form "vx_{i}.npy")
    vx_files = sorted([f for f in os.listdir(vx_dir) if f.endswith('.npy')])
    combined_images = []
    
    for vx_filename in vx_files:
        vx_path = os.path.join(vx_dir, vx_filename)
        
        # Extract the index from the vx filename.
        # For example, if vx_filename == "vx_0.npy", then index_str becomes "0".
        index_str = vx_filename.split('_')[-1].replace('.npy', '')
        
        # Construct the corresponding vy filename using the vy prefix.
        vy_filename = f"vy_{index_str}.npy"
        vy_path = os.path.join(vy_dir, vy_filename)
        
        # Load vx and vy arrays.
        vx_img = np.load(vx_path)
        vy_img = np.load(vy_path)
        
        # If the arrays are 2D, add a channel dimension.
        if vx_img.ndim == 2:
            vx_img = np.expand_dims(vx_img, axis=-1)
        if vy_img.ndim == 2:
            vy_img = np.expand_dims(vy_img, axis=-1)
        

        if third_channel is None:
            x_img = np.zeros_like(vx_img)
        else:
            x_filename = f"{third_channel}_{index_str}.npy"
            x_path = os.path.join(x_dir, x_filename)
            x_img = np.load(x_path)
            if x_img.ndim == 2:
                x_img = np.expand_dims(x_img, axis=-1)
                
        # Concatenate the channels to form a 3-channel image.
        combined = np.concatenate([vx_img, vy_img, x_img], axis=-1)
        combined_images.append(combined)
    
    return tf.convert_to_tensor(combined_images, dtype=tf.float32)


