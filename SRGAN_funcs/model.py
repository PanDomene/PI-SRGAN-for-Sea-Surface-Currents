import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from .losses import *


###############################################################
########################### MODEL #############################
###############################################################

# Pixel Shuffler
class PixelShuffle(layers.Layer):
    def __init__(self, block_size, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({
            "block_size": self.block_size
        })
        return config

# Generator: Residual Block
def residual_block(input_tensor, filters=64, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.add([input_tensor, x])

# Generator: Upscaling Block
def upsample_block(input_tensor, filters=64, scale=2):
    x = layers.Conv2D(filters * (scale ** 2), kernel_size=3, padding="same")(input_tensor)
    pixel_shuffle = PixelShuffle(scale)
    x = pixel_shuffle(x)
    return layers.PReLU(shared_axes=[1, 2])(x)

# Generator Network
def build_generator(input_shape=(16, 16, 3), num_residual_blocks=16):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, kernel_size=9, padding="same")(inputs)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    
    # Store the transformed inputs for the skip connection
    skip_connection = x

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x)
    
    # Add the skip connection
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, skip_connection])  # Ensure shapes are compatible here
    
    # Upsampling (use multiple upsample blocks for higher scaling)
    x = upsample_block(x, scale=2)
    x = upsample_block(x, scale=2)
    
    # Final output layer
    outputs = layers.Conv2D(3, kernel_size=9, activation="tanh", padding="same")(x)
    return Model(inputs, outputs, name="Generator")

## Discriminator Network
def build_discriminator(input_shape=(64, 64, 3), dropout_rates=(0.2, 0.3)):
    conv_dropout, connected_dropout = dropout_rates
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # Add more Conv2D + LeakyReLU blocks
    for filters, strides in [(64, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2)]:
        x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dropout(conv_dropout)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(connected_dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="Discriminator")

# Gaussian noise for better discriminator generalization
def add_input_noise(images, noise_std=0.05):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_std)
    return images + noise

# VGG Feature Extractor (for content loss)
def build_vgg_for_content_loss(input_shape=(None, None, 3)):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)
    model.trainable = False
    return model

def load_generator(path):
    generator = tf.keras.models.load_model(
        path,
        custom_objects={
            "PixelShuffle": PixelShuffle,
            "residual_block": residual_block,
            "upsample_block": upsample_block,
            "content_loss": content_loss,
            "discriminator_loss": discriminator_loss,
            "adversarial_loss": adversarial_loss
        },
        compile=True,
        safe_mode=False
    )
    return generator

def load_model(path, last=False, mse=False):
    # Reload the models
    if last:
        q = "_last.keras"
    elif mse:
        q = "_mse.keras"
    else:
        q = ".keras"
    
    generator = load_generator(f"{path}/generator{q}")
    discriminator = tf.keras.models.load_model(f"{path}/discriminator{q}")

    return generator, discriminator

