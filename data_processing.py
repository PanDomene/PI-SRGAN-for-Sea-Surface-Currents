import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import copernicusmarine as cm
import tensorflow as tf
from SRGAN_funcs import load_and_combine_channels
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
import joblib
import os


############### get path
path = os.getcwd()

############### load data
os.makedirs("data", exist_ok=True)

def download(start_date, end_date, file_name):
    
    try:
        data = netCDF4.Dataset(f'data/{file_name}.nc', 'r')
    except:
        cm.subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
            variables=["thetao", "uo", "vo", "zos"],
            minimum_longitude=-96.92524957370406, # west
            maximum_longitude=-90.69459354433525, # east
            minimum_latitude=20.825606179962158, # south
            maximum_latitude=26.883188430737402, # north
            start_datetime=start_date,
            end_datetime=end_date,
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            output_directory='data',
            output_filename=f"{file_name}.nc", 
            overwrite=True
        )
    
        data = netCDF4.Dataset(f'data/{file_name}.nc', 'r')

    return data

data = download("2022-06-01T12:00:00", "2025-06-01T12:00:00", "all_data")

############### Extract features
temp, vx, vy, ssh, _, lat, lng, _ = [np.array(data[var]) for var in data.variables.keys()]

# pack for easier manipulation
data = {"temp": temp, "vx": vx, "vy": vy, "ssh": ssh}

############# Temporal subsample: keep 1 measurement per day

# Downsample time
for var in data:
    arr = data[var][::24]
    arr = arr.squeeze(axis=1) # remove extra axis
    data[var] = arr


############# Data augmentation + croping

def get_crop_coords(t, h, w, crop_size=64, num_crops=1, seed=None):
    """
    Generate consistent random crop coordinates for each timestep.

    Args:
        t (int): Number of timesteps.
        h (int): Height (latitude) of the spatial grid.
        w (int): Width (longitude) of the spatial grid.
        crop_size (int, optional): Size of the square crop. Default is 64.
        num_crops (int, optional): Number of crops to generate per timestep. Default is 1.
        seed (int or None, optional): Random seed for reproducibility. Default is None.

    Returns:
        List[Tuple[int, int, int]]: A list of (timestep_index, lat_start, lon_start) tuples
                                    specifying the crop positions.
    """
    rng = np.random.default_rng(seed) # random number generator
    coords = []

    for i in range(t): # for each timestep
        for _ in range(num_crops): # if num_crops>1, data augmentation takes place
            lat_start = rng.integers(0, h - crop_size + 1)
            lon_start = rng.integers(0, w - crop_size + 1)
            coords.append((i, lat_start, lon_start)) 
    
    return coords

def apply_crops(arr, coords, crop_size=64):
    """
    Apply a list of crop coordinates to a 3D array (time, height, width).

    Args:
        arr (np.ndarray): Input array of shape (t, h, w) to crop.
        coords (List[Tuple[int, int, int]]): List of crop coordinates 
            in the form (timestep_index, lat_start, lon_start).
        crop_size (int, optional): Size of the square crop. Default is 64.

    Returns:
        np.ndarray: Cropped array of shape (len(coords), crop_size, crop_size).
    """
    crops = []
    # if data augmentation was performed, 
    # there will be multiple coords with the same i.
    for i, lat_start, lon_start in coords:
        crop = arr[i, lat_start:lat_start + crop_size, lon_start:lon_start + crop_size]
        crops.append(crop)
    return np.stack(crops)

# Step 1: Split full data
t, h, w = data["temp"].shape
train_time = 365
val_time = train_time + 365 // 2

split_data = {}
for var in data:
    split_data[f"{var}_train"] = data[var][:train_time]
    split_data[f"{var}_val"] = data[var][train_time:val_time]
    split_data[f"{var}_test"] = data[var][val_time:]

# Step 2: Generate crop coordinates (all random, but only train has >1 per t)
crop_size = 64

train_coords = get_crop_coords(
    t=split_data["temp_train"].shape[0],
    h=h, w=w,
    crop_size=crop_size,
    num_crops=6,  # data augmentation
    seed=42
)

val_coords = get_crop_coords(
    t=split_data["temp_val"].shape[0],
    h=h, w=w,
    crop_size=crop_size,
    num_crops=1,
    seed=43  # different seed to get new (but consistent) crops
)

test_coords = get_crop_coords(
    t=split_data["temp_test"].shape[0],
    h=h, w=w,
    crop_size=crop_size,
    num_crops=1,
    seed=44
)


# Step 3: Apply crops to all variables
for var in ['temp', 'vx', 'vy', 'ssh']:
    split_data[f"{var}_train"] = apply_crops(split_data[f"{var}_train"], train_coords, crop_size)
    split_data[f"{var}_val"] = apply_crops(split_data[f"{var}_val"], val_coords, crop_size)
    split_data[f"{var}_test"] = apply_crops(split_data[f"{var}_test"], test_coords, crop_size)


# Step 4: Standardize (fit only on training data)
scalers = {}

for var in ['temp', 'vx', 'vy', 'ssh']:
    arr = split_data[f"{var}_train"]
    scaler = StandardScaler()
    scaler.fit(arr.reshape(-1, 1))
    scalers[var] = scaler

for key in split_data:
    var = key.split('_')[0]
    arr = split_data[key]
    n, h, w = arr.shape
    split_data[key] = scalers[var].transform(arr.reshape(-1, 1)).reshape(n, h, w)


# Step 5: Save scalers
os.makedirs("scalers", exist_ok=True)
for var, scaler in scalers.items():
    joblib.dump(scaler, f"scalers/{var}.pkl")


################ Gaussian blur + spatial subsampling

def blur_and_downsample(arr, sigma=1.0, factor=4):
    """
    Apply Gaussian blur and downsample by a factor to each 2D image.

    Parameters:
        arr: np.ndarray of shape (n, h, w)
        sigma: float or sequence, standard deviation for Gaussian blur
        factor: int, downsampling factor (must divide h and w evenly)

    Returns:
        low_res: np.ndarray of shape (n, h//factor, w//factor)
    """
    n, h, w = arr.shape
    assert h % factor == 0 and w % factor == 0, "Size must be divisible by factor."

    blurred = np.empty_like(arr)
    for i in range(n):
        blurred[i] = gaussian_filter(arr[i], sigma=sigma)

    # Downsample by taking every `factor` pixel
    low_res = blurred[:, ::factor, ::factor]
    return low_res


split_data_lr = {}

for key, value in split_data.items():
    split_data_lr[key] = blur_and_downsample(value, sigma=1.0, factor=4)


############ Save data

path = os.path.join(os.getcwd(), "data")

center = len(lat) // 2
half = 64 // 2
lat = lat[center - half : center + half]

center = len(lng) // 2
half = 64 // 2
lng = lng[center - half : center + half]

np.save(f"{path}/lat.npy", lat)
np.save(f"{path}/lng.npy", lng)

def save_data(data, path, prefix):
    os.makedirs(path, exist_ok=True)
    for i in range(data.shape[0]):
        filename = os.path.join(path, f"{prefix}_{i}.npy")
        np.save(filename, data[i])

for key, val in split_data.items():
    var, split = key.split("_")
    save_data(val, f"{path}/{split}/HR/{var}/", var)

for key, val in split_data_lr.items():
    var, split = key.split("_")
    save_data(val, f"{path}/{split}/LR/{var}/", var)