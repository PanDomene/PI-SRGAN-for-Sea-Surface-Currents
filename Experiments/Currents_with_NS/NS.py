import gc
from SRGAN_funcs import *
import tensorflow as tf
import os

# Garbage collection (clears Python-side objects)
gc.collect()

# Clear TensorFlow's internal session and GPU memory
tf.keras.backend.clear_session()

# Optional: Enable dynamic memory growth on GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")


###############################################################
########################### PARAMETERS ########################
###############################################################


hr_size = 64
lr_size = hr_size//4
R = 6378000.0  # Earth radius in meters
pi = tf.constant(np.pi, dtype=tf.float32)
deg_to_rad = pi / 180.0
g = 9.81 # Gravity
eta = 0 # Viscosity
laplace = False
advection = False
div_weight = 1e-2


if __name__ == "__main__":
    folder_name, initial_lr, max_lr, batch_size, epochs, gen_adv_weight, noise_std, conv_dropout, connected_dropout = main()
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    warmup_epochs = int(0.1 * epochs)

###############################################################
########################### MODEL #############################
###############################################################

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.9)

# Model
generator, discriminator = load_model("../currents_and_SSH/models/01/", mse=True)
vgg = build_vgg_for_content_loss()
################################################################
################### COMPUTE GRID SPACINGS ######################
################################################################

# Centered spacing between grid points + Coriolis parameter
dx, dy, f = compute_grid_spacing()

################################################################
######################### LOAD DATA ############################
################################################################

# Define folder paths for training HR and LR data.
train_hr_folder = os.path.join(os.getcwd(), "../data", "train", "HR")
train_lr_folder = os.path.join(os.getcwd(), "../data", "train", "LR")
val_hr_folder = os.path.join(os.getcwd(), "../data", "val", "HR")
val_lr_folder = os.path.join(os.getcwd(), "../data", "val", "LR")

# Load and combine channels for HR and LR images.
hr_images = load_and_combine_channels(train_hr_folder, "ssh")
lr_images = load_and_combine_channels(train_lr_folder, "ssh")
val_hr = load_and_combine_channels(val_hr_folder, "ssh")
val_lr = load_and_combine_channels(val_lr_folder, "ssh")

# Create a TensorFlow dataset.
dataset = tf.data.Dataset.from_tensor_slices((lr_images, hr_images))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

means, stds = load_means_and_stds("../../scalers")

##############################################################################
################################ Training Loop ###############################
##############################################################################

train_NS(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset, val_lr, val_hr, f, dx, dy, epochs, warmup_epochs, initial_lr, max_lr, gen_adv_weight, folder_name, laplace, advection, div_weight, means, stds)
