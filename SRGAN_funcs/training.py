from .model import *
from .losses import *
from .physics import *
import argparse
import csv

# Learning Rate Schedule

def get_lr(epochs, epoch, warmup_epochs, initial_lr, max_lr, generator_optimizer):
    if epoch < warmup_epochs:
        return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs) # Linear warm-up
    elif epoch < 0.7 * epochs:
        return generator_optimizer.learning_rate.value
    else: # Cosine decay
        return max_lr * 0.5 * (1 + tf.math.cos(((epoch - warmup_epochs) / (epochs - warmup_epochs)) * tf.constant(3.1415926)))

# For standardization-related issues

def destandardize(tensor, stds, means, keys=("vx", "vy", "ssh")):
    """
    De-standardizes a tensor of shape (..., 3) using provided std and mean dicts.
    """
    return tf.stack([
        tensor[..., i] * stds[key] + means[key]
        for i, key in enumerate(keys)
    ], axis=-1)


import os
import joblib  # or use pickle if you used that
from sklearn.preprocessing import StandardScaler

def load_means_and_stds(scaler_dir):
    """
    Loads StandardScalers from a directory and returns two dictionaries:
    one for means and one for standard deviations.

    Args:
        scaler_dir (str): Directory containing .pkl scaler files (one per variable).

    Returns:
        means (dict[str, float]): Mean values for each variable.
        stds (dict[str, float]): Standard deviation values for each variable.
    """
    means = {}
    stds = {}

    for fname in os.listdir(scaler_dir):
        if fname.endswith(".pkl"):
            var = fname.replace(".pkl", "")
            scaler_path = os.path.join(scaler_dir, fname)
            scaler = joblib.load(scaler_path)

            means[var] = float(scaler.mean_)
            stds[var] = float(scaler.scale_)

    return means, stds

################################
########## DATA ONLY ###########
################################


@tf.function
def train_step(generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, lr, hr, gen_adv_weight):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr = generator(lr, training=True)
        
        hr_noisy = add_input_noise(hr)
        sr_noisy = add_input_noise(sr)

        real_output = discriminator(hr_noisy, training=True)
        fake_output = discriminator(sr_noisy, training=True)
        
        ##### Losses #####
        con_loss = content_loss(vgg, hr, sr)
        gen_adv_loss = adversarial_loss(fake_output)

        gen_loss = con_loss + gen_adv_weight * gen_adv_loss
        disc_loss = discriminator_loss(real_output, fake_output)
                
    # Gradients and Optimizer
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, con_loss, gen_adv_loss


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset, val_lr, val_hr, epochs=100, warmup_epochs=10, initial_lr=1e-6, max_lr=1e-3, gen_adv_weight=1e-3, folder_name=None, means=None, stds=None):

    assert means is not None and stds is not None, "Both 'means' and 'stds' must be provided to compute physical MSE."
    
    vgg = build_vgg_for_content_loss()

    # Initialize best (generator loss) and best_mse for saving checkpoints
    best = np.inf
    best_mse = np.inf
    
    # Open files to write learning curves & loss term values 
    losses_file = open(f'models/{folder_name}/losses.csv', mode='w', newline='')
    loss_writer = csv.writer(losses_file)
    loss_writer.writerow(["gen_loss", "disc_loss", "con_loss", "gen_adv_loss"])

    curves_file = open(f'models/{folder_name}/learning_curves.csv', mode='w', newline='')
    curve_writer = csv.writer(curves_file)
    curve_writer.writerow(["gen_train", "disc_train", "gen_val", "disc_val", "best", "val_mse", "best_mse"])
    batch_idx = 0
    # Training Loop
    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        batch_count = 0
            
        # Manually update the learning rate
        new_lr = get_lr(epochs, epoch, warmup_epochs, initial_lr, max_lr, generator_optimizer)
        generator_optimizer.learning_rate.assign(new_lr)
        discriminator_optimizer.learning_rate.assign(new_lr)

        for low_res, high_res in dataset:
            
            gen_loss, disc_loss, con_loss, gen_adv_loss = train_step(generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, low_res, high_res, gen_adv_weight)
                
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            batch_count += 1

            # Save losses for debugging
            loss_writer.writerow([
                float(gen_loss), 
                float(disc_loss), 
                float(con_loss), 
                float(gen_adv_weight*gen_adv_loss)
            ])
            if batch_idx % 5 == 0:
                # Validation
                val_sample_size = 128
                idx = np.random.choice(len(val_lr), size=val_sample_size, replace=False)

                val_lr_batch = tf.gather(val_lr, idx)
                val_hr_batch = tf.gather(val_hr, idx)
                
                # Inference and validation
                sr = generator(val_lr_batch, training=False)
                sr_score = discriminator(sr, training=False)
                hr_score = discriminator(val_hr_batch, training=False)
                con_loss = content_loss(vgg, val_hr_batch, sr)
                gen_adv_loss = adversarial_loss(sr_score)
    
                val_gen_loss = con_loss + gen_adv_weight * gen_adv_loss
                val_disc_loss = discriminator_loss(hr_score, sr_score)
                
                # Save if generator loss in validation went down
                if val_gen_loss < best:
                    generator.save(f"models/{folder_name}/generator.keras")
                    discriminator.save(f"models/{folder_name}/discriminator.keras")
                    best = val_gen_loss
                    print(f"model saved (generator loss): {best}")
    
                # De-standardize
                sr_phys = destandardize(sr, stds, means)
                hr_phys = destandardize(val_hr_batch, stds, means)
    
                # Compute MSE in physical units (vx and vy only)
                squared_diff = tf.square(hr_phys[..., :2] - sr_phys[..., :2])
                mse = tf.reduce_mean(squared_diff)
    
                # Save if validation MSE went down
                if mse < best_mse:
                    generator.save(f"models/{folder_name}/generator_mse.keras")
                    discriminator.save(f"models/{folder_name}/discriminator_mse.keras")
                    best_mse = mse
                    print(f"model saved (MSE): {mse}")
    
                # update learning curves
                curve_writer.writerow([
                    float(gen_loss), 
                    float(disc_loss),
                    float(val_gen_loss), 
                    float(val_disc_loss),
                    float(best), 
                    float(mse), 
                    float(best_mse)
                ])

            batch_idx += 1
            
        avg_gen_loss = epoch_gen_loss / batch_count
        avg_disc_loss = epoch_disc_loss / batch_count
        
        
        print(f"[Epoch {epoch + 1:03d}] Avg Gen: {avg_gen_loss:.4f}, Avg Disc: {avg_disc_loss:.4f}, Val Gen: {val_gen_loss:.4f}, Val MSE: {mse:.6f}, LR: {generator_optimizer.learning_rate.numpy():.6f}")

    # Close CSV writers
    losses_file.close()
    curves_file.close()

    # Save the final model after training
    generator.save(f"models/{folder_name}/generator_last.keras")
    discriminator.save(f"models/{folder_name}/discriminator_last.keras")



################################
####### Data + Physics #########
################################


@tf.function
def train_step_NS(generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, lr, hr, w, gen_adv_weight, f, dx, dy, laplace, advection, div_weight, means, stds):
    """w determines the importance of the content vs physical losses."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr = generator(lr, training=True)
        
        hr_noisy = add_input_noise(hr)
        sr_noisy = add_input_noise(sr)

        real_output = discriminator(hr_noisy, training=True)
        fake_output = discriminator(sr_noisy, training=True)
        
        ##### Losses #####
        con_loss = content_loss(vgg, hr, sr)
        gen_adv_loss = adversarial_loss(fake_output)

        gen_loss = con_loss + gen_adv_weight * gen_adv_loss
        disc_loss = discriminator_loss(real_output, fake_output)
        
        ### Physical Losses ###
        ns_loss = navier_stokes_loss(sr, f, laplace, advection, dx, dy, means, stds)
        div_loss = divergence_loss(sr, dx, dy, means, stds)
        
        gen_loss =  (1 - w) * gen_loss + w * (ns_loss + div_weight * div_loss)
        
    # Gradients and Optimizer
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, ns_loss, con_loss, gen_adv_loss, div_loss



def train_NS(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset, val_lr, val_hr, f, dx, dy,
          epochs=100, warmup_epochs=10, initial_lr=1e-6, max_lr=1e-3, gen_adv_weight=1e-3, folder_name=None, laplace=False, advection=False, div_weight=1, means=None, stds=None):

    assert means is not None and stds is not None, "Both 'means' and 'stds' must be provided to compute physical MSE."
    
    vgg = build_vgg_for_content_loss()

    # Find scores for the pretrained model
    sr = generator(val_lr)
    
    hr_score = discriminator(val_hr, training=False)
    sr_score = discriminator(sr, training=False)
    
    con_loss = content_loss(vgg, val_hr, sr)
    gen_adv_loss = adversarial_loss(sr_score)
    gen_loss = con_loss + gen_adv_weight * gen_adv_loss
    best = gen_loss
    
    # De-standardize both sr and val_hr
    sr_phys = destandardize(sr, stds, means)
    hr_phys = destandardize(val_hr, stds, means)
    
    # Compute MSE in physical units (vx and vy only)
    squared_diff = tf.square(hr_phys[..., :2] - sr_phys[..., :2])
    best_mse = tf.reduce_mean(squared_diff)

    # Start with no physical constraints
    w = tf.Variable(0.0, dtype=tf.float32) 
    # Increase of importance of physical part per epoch
    dw = tf.constant(3e-4, dtype=tf.float32) 

    losses_file = open(f'models/{folder_name}/losses.csv', mode='w', newline='')
    loss_writer = csv.writer(losses_file)
    loss_writer.writerow(["gen_loss", "disc_loss", "ns_loss", "con_loss", "gen_adv_loss", "div_loss"])

    curves_file = open(f'models/{folder_name}/learning_curves.csv', mode='w', newline='')
    curve_writer = csv.writer(curves_file)
    curve_writer.writerow(["gen_train", "disc_train", "gen_val", "disc_val", "best", "val_mse", "best_mse"])

    # Log raw loss terms (unweighted) for analysis
    raw_losses_file = open(f'models/{folder_name}/raw_losses.csv', mode='w', newline='')
    raw_loss_writer = csv.writer(raw_losses_file)
    raw_loss_writer.writerow(["con_loss", "gen_adv_loss", "ns_loss", "div_loss", "gen_loss", "disc_loss", "w"])

    batch_idx = 0
    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        batch_count = 0
            
        # Manually update the learning rate
        new_lr = get_lr(epochs, epoch, warmup_epochs, initial_lr, max_lr, generator_optimizer)
        generator_optimizer.learning_rate.assign(new_lr)
        discriminator_optimizer.learning_rate.assign(new_lr)

        for low_res, high_res in dataset:
            
            gen_loss, disc_loss, ns_loss, con_loss, gen_adv_loss, div_loss = train_step_NS(generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, low_res, high_res, w, gen_adv_weight, f, dx, dy, laplace, advection, div_weight, means, stds)

            raw_loss_writer.writerow([
                float(con_loss),
                float(gen_adv_loss),
                float(ns_loss),
                float(div_loss),
                float(gen_loss),
                float(disc_loss),
                float(w.numpy())
            ])
            
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            batch_count += 1

            # Save losses for debugging
            loss_writer.writerow([
                float((1-w)*gen_loss), 
                float(disc_loss), 
                float(w*ns_loss), 
                float(con_loss), 
                float(gen_adv_weight*gen_adv_loss), 
                float(w*div_weight*div_loss)
            ])

            if batch_idx % 5 == 0:
                # Validation
                val_sample_size = 128
                idx = np.random.choice(len(val_lr), size=val_sample_size, replace=False)

                val_lr_batch = tf.gather(val_lr, idx)
                val_hr_batch = tf.gather(val_hr, idx)
                
                # Inference and validation
                sr = generator(val_lr_batch, training=False)
                sr_score = discriminator(sr, training=False)
                hr_score = discriminator(val_hr_batch, training=False)
                con_loss = content_loss(vgg, val_hr_batch, sr)
                gen_adv_loss = adversarial_loss(sr_score)
    
                val_gen_loss = con_loss + gen_adv_weight * gen_adv_loss
                val_disc_loss = discriminator_loss(hr_score, sr_score)
                
                # Save if generator loss in validation went down
                if val_gen_loss < best:
                    generator.save(f"models/{folder_name}/generator.keras")
                    discriminator.save(f"models/{folder_name}/discriminator.keras")
                    best = val_gen_loss
                    print(f"model saved (generator loss): {best}")
    
                
                # De-standardize both sr and val_hr
                sr_phys = destandardize(sr, stds, means)
                hr_phys = destandardize(val_hr, stds, means)
                
                # Compute MSE in physical units (vx and vy only)
                squared_diff = tf.square(hr_phys[..., :2] - sr_phys[..., :2])
                mse = tf.reduce_mean(squared_diff)
    
                # Save if validation MSE went down
                if mse < best_mse:
                    generator.save(f"models/{folder_name}/generator_mse.keras")
                    discriminator.save(f"models/{folder_name}/discriminator_mse.keras")
                    best_mse = mse
                    print(f"model saved (MSE): {mse}")
    
                # update learning curves
                curve_writer.writerow([
                    float(gen_loss), 
                    float(disc_loss),
                    float(val_gen_loss), 
                    float(val_disc_loss),
                    float(best), 
                    float(mse), 
                    float(best_mse)
                ])
            batch_idx += 1
            
        avg_gen_loss = epoch_gen_loss / batch_count
        avg_disc_loss = epoch_disc_loss / batch_count
        # Update Physical weight
        w.assign_add(dw)
        
        print(f"[Epoch {epoch + 1:03d}] Avg Gen: {avg_gen_loss:.4f}, Avg Disc: {avg_disc_loss:.4f}, Val Gen: {val_gen_loss:.4f}, Val MSE: {mse:.6f}, LR: {generator_optimizer.learning_rate.numpy():.6f}")

    # Close CSV writers
    losses_file.close()
    curves_file.close()
    raw_losses_file.close()
    
    # Save the final model after training
    generator.save(f"models/{folder_name}/generator_last.keras")
    discriminator.save(f"models/{folder_name}/discriminator_last.keras")



############ Parameter "automation" ################

def main():
    parser = argparse.ArgumentParser(description="Train SRGAN model with specified parameters.")
    parser.add_argument('--folder_name', type=str, default="test", help='Folder name for the experiment')
    parser.add_argument('--initial_lr', type=float, default=0.000001, help='Initial Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.0001, help='Max Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gen_adv_weight', type=float, default=1e-3, help='Weight for the Generator Adversarial Loss')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Std for the noise added to the input of the discriminator during training')
    parser.add_argument('--conv_dropout', type=float, default=0.2, help='Dropout rate for convolutions in discriminator')
    parser.add_argument('--connected_dropout', type=float, default=0.3, help='Dropout rate for fully connected part in discriminator')
    args = parser.parse_args()

    # Access the arguments using args.lr, args.batch_size, and args.epochs
    print(f"Initial learning rate: {args.initial_lr}")
    print(f"Max learning rate: {args.max_lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"gen_adv_weight: {args.gen_adv_weight}")

    return args.folder_name, args.initial_lr, args.max_lr, args.batch_size, args.epochs, args.gen_adv_weight, args.noise_std, args.conv_dropout, args.connected_dropout
