import gc
from SRGAN_funcs import *
import tensorflow as tf
import joblib
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
generator = build_generator()
discriminator = build_discriminator()

################################################################
######################### LOAD DATA ############################
################################################################

# Define folder paths for training HR and LR data.
train_hr_folder = os.path.join(os.getcwd(), "../data", "train", "HR")
train_lr_folder = os.path.join(os.getcwd(), "../data", "train", "LR")
val_hr_folder = os.path.join(os.getcwd(), "../data", "val", "HR")
val_lr_folder = os.path.join(os.getcwd(), "../data", "val", "LR")

# Load and combine channels for HR and LR images.
hr_images = load_and_combine_channels(train_hr_folder)
lr_images = load_and_combine_channels(train_lr_folder)
val_hr = load_and_combine_channels(val_hr_folder)
val_lr = load_and_combine_channels(val_lr_folder)

# Create a TensorFlow dataset.
dataset = tf.data.Dataset.from_tensor_slices((lr_images, hr_images))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

means, stds = load_means_and_stds("../../scalers")

##############################################################################
################################ Training Loop ###############################
##############################################################################

train(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset, val_lr, 
      val_hr, epochs, warmup_epochs, initial_lr, max_lr, gen_adv_weight, folder_name, means, stds)