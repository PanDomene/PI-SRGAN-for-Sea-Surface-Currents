import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.preprocessing import image
from matplotlib.patches import ConnectionPatch
from skimage.metrics import structural_similarity as ssim
from .data_utils import load_and_combine_channels
from .model import load_model

hr_size = 64
lr_size = hr_size//4

def learning_curves(path, height_loss=0.8, width_loss=2.5, height_mse=0.2, width_mse=0.8):
    
    # Create a path for figures inside the 'reports' folder.
    splits = path.split("/")
    path_figs = splits[-3] + "/" + splits[-1]
    path_figs = f'E:/Documents/MING/Tesis/Proyecto/reports/figures/{path_figs}'
    os.makedirs(path_figs, exist_ok=True)
    
    lc = pd.read_csv(f"{path}/learning_curves.csv") #lc for "learning curves"
    lc.columns = ["Generator loss (train data)", "Discriminator loss (train data)", 
                  "Generator loss (validation data)", "Discriminator loss (validation data)",
                  "Lowest validation loss", "Validation MSE", "Lowest validation MSE"]

    batches = 737 // 64 + 1 # 737: size of training set, 64: batch size
    lc.iloc[:, 0:4] = lc.iloc[:, 0:4].rolling(window=batches).mean() # Get the averages per epoch
    lc.iloc[:, 5] = lc.iloc[:, 5].rolling(window=batches).mean() 


    i_best = np.argmin(lc.iloc[:, 4]) // batches # Epoch of best loss
    best_loss = lc.iloc[:, 4].min() # Best loss value
    
    i_mse = np.argmin(lc.iloc[:, 6]) // batches # Epoch of best MSE
    best_mse = lc.iloc[:, 6].min() # Best mse value
    
    y_max =  max(lc.iloc[:, [2, 5]].max()) # For the vertical lines.
    
    lc = lc.iloc[::batches] # Keep the average for each epoch
    lc.index = range(len(lc)) # Change the index to epoch num. instead of iteration. 
    
    ###### learning curves and metrics #######
    fig, ax = plt.subplots(3, 1, figsize=(7, 9))
    
    lc.iloc[:, [0, 2]].plot(ax = ax[0], linewidth=0.8)
    ax[0].legend(['Training', 'Validation'])
    ax[0].set_yscale('log')

    lc.iloc[:, [1, 3]].plot(ax = ax[1], linewidth=0.8)
    ax[1].set_yscale('log')

    lc.iloc[:, 5].plot(ax = ax[2], color='slategray', alpha=0.5, label="Validation MSE", linewidth=0.8)
    lc.iloc[:, [4, 6]].plot(ax = ax[2], color=['dodgerblue', 'limegreen'], linewidth=0.8)
    ax[2].legend()
    ax[2].vlines(i_best, best_loss, y_max, "dodgerblue", ":")
    ax[2].vlines(i_mse, best_mse, y_max, "limegreen", ":")
    ax[2].text(width_loss * i_best, height_loss * y_max, 
             f"Smallest Loss found\n on {i_best}'th epoch,\nLoss={round(best_loss, 4)}", 
             color='dodgerblue', ha='center', va='top')
    ax[2].text(width_mse * i_mse, height_mse * y_max, 
             f"Smallest MSE found\n on {i_mse}'th epoch,\nMSE={round(best_mse, 4)}", 
             color='limegreen', ha='center', va='top')
    
    ax[2].set_yscale('log')
    ax[0].set_title('Generator Learning Curves')
    ax[1].set_title('Discrimintor Learning Curves')
    ax[2].set_title('Metrics')
    
    fig.tight_layout()
    
    fig.savefig(f"{path_figs}/learning_curves.jpg", dpi=300)
    plt.show()
    plt.close()
    
    ######## Different loss terms ############

    losses = pd.read_csv(f"{path}/losses.csv")

    losses = losses.rolling(window=batches).mean()
    
    losses = losses.iloc[::batches, [0, 2, 3]]
    losses.index = range(len(losses))
    losses.columns = ["Generator Loss (a + b)", "Content Loss (a)", "Generator Adversarial Loss (b)"]
    
    fig, ax = plt.subplots(figsize=(6, 4))

    losses.plot(ax=ax, linewidth=0.8)  # now plotting directly on your `fig`
    ax.set_yscale("log")
    ax.set_title("Training Losses")
    
    fig.tight_layout()
    fig.savefig(f"{path_figs}/generator_term_values.jpg", dpi=300)
    plt.show()
    plt.close()



def get_SSIM(true, preds):
    if isinstance(true, tf.Tensor):
        true = true.numpy()
        preds = preds.numpy()

    ssims = []
    for i in range(len(true)):
        hr, pred = true[i], preds[i]

        # Ensure both images have the same shape
        if hr.shape != pred.shape:
            pred = np.reshape(pred, hr.shape)  # Force the same shape

        max_ = max(hr.max(), pred.max())
        min_ = min(hr.min(), pred.min())
        data_range = max_ - min_

        is_rgb = hr.ndim == 3 and hr.shape[-1] == 3  # Check if it's RGB
        ssims.append(ssim(hr, pred, channel_axis=-1 if is_rgb else None, data_range=data_range))

    return np.mean(ssims)


def get_RMSE(true, preds):
    
    squared_diff = tf.square(true - preds)
    mse = tf.reduce_mean(squared_diff)
    rmse = tf.sqrt(mse)
    
    return rmse.numpy()  # Convert to NumPy for easy use


def normalize_batch(*image_sets):
    """Normalize all images in the batch to [0, 1] using the global min/max."""
    global_min = min(np.min(images) for images in image_sets)
    global_max = max(np.max(images) for images in image_sets)
    
    def normalize(images):
        return (images - global_min) / (global_max - global_min) if global_max > global_min else images

    return tuple(normalize(images) for images in image_sets)


def bicubic_resize(image, target_height=hr_size, target_width=hr_size):
    return tf.image.resize(image, [target_height, target_width], method=tf.image.ResizeMethod.BICUBIC)


def evaluate_images(true, predicted, bicubic=None):
    """Calculate quality metrics between predicted/bicubic and ground truth."""
    results = {
        'sr_ssim': get_SSIM(true, predicted),
        'sr_rmse': get_RMSE(true, predicted)
    }
    
    if bicubic is not None:
        results['bicubic_ssim'] = get_SSIM(true, bicubic)
        results['bicubic_rmse'] = get_RMSE(true, bicubic)
    
    return results


def visualize_batch(low_res, hr, sr, bicubic, title=None, rows=4, speed=False, field=False):

    low_res = low_res[:rows]
    hr = hr[:rows]
    sr = sr[:rows]
    bicubic = bicubic[:rows]
    
    if field:
        d = 3
        x_arrows = [low_res[:, :, :, 0]] + [images[:, ::d, ::d, 0] for images in [bicubic, sr, hr]]
        y_arrows = [low_res[:, :, :, 1]] + [images[:, ::d, ::d, 1] for images in [bicubic, sr, hr]]
        xy_hr = np.arange(0, hr.shape[1], d)
        xy_lr = np.arange(0, low_res.shape[1])
    if speed:        
        hr = tf.norm(hr, axis=-1) # axis=-1 so that the norm is taken over the 3 velocity components.
        low_res = tf.norm(low_res, axis=-1)
        sr = tf.norm(sr, axis=-1)
        bicubic = tf.norm(bicubic, axis=-1)
        
    # _n for normalized
    hr_n, low_res_n, sr_n, bicubic_n = normalize_batch(hr, low_res, sr, bicubic)
    
    # Create figure with space for colorbar
    if speed:
        fig = plt.figure(figsize=(10, 2.5 * rows))  # Make figure slightly wider for colorbar
        gs = fig.add_gridspec(rows, 5, width_ratios=[1, 1, 1, 1, 0.1])  # 5th column for colorbar
    else:
        fig = plt.figure(figsize=(9, 2.5 * rows))
        gs = fig.add_gridspec(rows, 4)
    
    channels = 3 if hr.shape[-1] == 3 else 1  # Determine if images are RGB or grayscale
    image_types = [low_res_n, bicubic_n, sr_n, hr_n]
    labels = ["Baja Resolución", "Interpolación Bicúbica", "SRGAN", "Original (HR)"]
    
    # Create variable to store the mappable for colorbar
    colorbar_mappable = None
    
    for row in range(rows):
        for col, (images, label) in enumerate(zip(image_types, labels)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(images[row], vmin=0, vmax=1)
            
            # Store the last mappable for the colorbar
            if row == 0 and col == 0:
                colorbar_mappable = im
                
            ax.set_aspect('auto')  # Fix aspect ratio
            
            if field:
                if col == 0:  # Low Resolution
                    plt.quiver(xy_lr, xy_lr, x_arrows[col][row], y_arrows[col][row],
                               scale=13, color='white', width=0.0035)
                else:  # Other image types
                    plt.quiver(xy_hr, xy_hr, x_arrows[col][row], y_arrows[col][row],
                               scale=13, color='white', width=0.0035)
                
            if row == 0:
                ax.set_title(label, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    # Add a single colorbar if speed is True
    if speed:
        cbar_ax = fig.add_subplot(gs[:, 4])  # Use the 5th column for colorbar
        cbar = fig.colorbar(colorbar_mappable, cax=cbar_ax)
        cbar.set_label('Rapidez Normalizada')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_zoom(bc, sr, hr, idx, coords_list, colors=['red', 'blue'], field=False):
    """
    Plots images and zooms into specified regions with different colored spines and field vectors.
    Parameters:
        bc, sr, hr: Arrays of images or tensors (shape: [batch, H, W, C])
        idx: Index of the image to extract from bc, sr, hr
        coords_list: List of coordinate sets, e.g., [[x1_start, x1_end, y1_start, y1_end], 
                                                      [x2_start, x2_end, y2_start, y2_end]]
        colors: List of colors for the spines (default: ['red', 'blue'])
        field: Boolean, whether to show vector field with quiver plots
    """
    import matplotlib.patches as patches
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    
    # Store original data for field vectors if needed
    if field:
        # Create numpy copies or convert tensors to numpy for vector fields
        bc_orig = bc.numpy() if hasattr(bc, 'numpy') else np.array(bc)
        sr_orig = sr.numpy() if hasattr(sr, 'numpy') else np.array(sr)
        hr_orig = hr.numpy() if hasattr(hr, 'numpy') else np.array(hr)
        
        # Default sampling density
        d = 3
        # Extract vector components for quiver plots (correct indexing)
        x_arrows = [images[idx, :, :, 0] for images in [bc_orig, sr_orig, hr_orig]]
        y_arrows = [images[idx, :, :, 1] for images in [bc_orig, sr_orig, hr_orig]]
        xy = np.arange(0, hr_orig.shape[1], d)
    
    # Convert to magnitude for visualization
    hr_mag = tf.norm(hr, axis=-1)  # axis=-1 so that the norm is taken over the 3 velocity components.
    sr_mag = tf.norm(sr, axis=-1)
    bc_mag = tf.norm(bc, axis=-1)
    
    # Convert tensors to numpy if needed
    hr_np = hr_mag.numpy() if hasattr(hr_mag, 'numpy') else np.array(hr_mag)
    sr_np = sr_mag.numpy() if hasattr(sr_mag, 'numpy') else np.array(sr_mag)
    bc_np = bc_mag.numpy() if hasattr(bc_mag, 'numpy') else np.array(bc_mag)
    
    assert len(coords_list) == len(colors), "Provide as many colors as there are zoom regions."
    
    # Extract images from batch
    bc_img, sr_img, hr_img = bc_np[idx], sr_np[idx], hr_np[idx]
    
    # Create figure with very tight spacing
    rows = 1 + len(coords_list)
    fig, axs = plt.subplots(rows, 3, figsize=(6, 6), 
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    # Make sure axs is 2D even if rows=1
    if rows == 1:
        axs = np.array([axs])
    
    # Titles for the columns
    titles = ["Interpolación Bicúbica", "SRGAN", "Original (HR)"]
    
    # Plot the original images with different colored rectangles
    for i, img in enumerate([bc_img, sr_img, hr_img]):
        axs[0, i].imshow(img)
        axs[0, i].set_title(titles[i], fontsize=11, pad=2)
        axs[0, i].axis("off")
        
        # Draw rectangles with different colors
        for coords, color in zip(coords_list, colors):
            x_start, x_end, y_start, y_end = coords
            rect = patches.Rectangle((y_start, x_start), y_end - y_start, x_end - x_start, 
                                     linewidth=1, edgecolor=color, facecolor='none')
            axs[0, i].add_patch(rect)
        
        # Add quiver to original images if field=True
        if field:
            # Use appropriate sampling for the full image
            axs[0, i].quiver(xy, xy, x_arrows[i][::d, ::d], y_arrows[i][::d, ::d], 
                            scale=11, color='white', width=0.003)
    
    # Plot zoomed-in regions with colored spines
    for row_idx, (coords, color) in enumerate(zip(coords_list, colors), start=1):
        x_start, x_end, y_start, y_end = coords
        
        for i, img in enumerate([bc_img, sr_img, hr_img]):
            # Get zoomed image for visualization
            zoomed_img = img[x_start:x_end, y_start:y_end]
            axs[row_idx, i].imshow(zoomed_img)
            axs[row_idx, i].axis("on")  # Ensure axis is on to show spines
            
            # Set spine colors and linewidth
            for spine in axs[row_idx, i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            
            # Hide tick labels and tick marks
            axs[row_idx, i].set_xticklabels([])
            axs[row_idx, i].set_yticklabels([])
            axs[row_idx, i].tick_params(axis='both', which='both', length=0)
            
            # Add quiver for zoomed region if field=True
            if field:
                # Extract vectors for just this region
                zoomed_x = x_arrows[i][x_start:x_end, y_start:y_end]
                zoomed_y = y_arrows[i][x_start:x_end, y_start:y_end]
                
                # Calculate appropriate sampling density based on zoom size
                # Smaller regions should have finer sampling
                region_size = min(x_end - x_start, y_end - y_start)
                zoom_d = max(1, int(d * 3 / region_size))  # Adjust sampling based on region size
                
                # Create coordinate grid for zoomed region
                zoomed_xy = np.arange(0, zoomed_img.shape[0], zoom_d)
                
                if len(zoomed_xy) > 0:  # Ensure we have points to plot
                    X, Y = np.meshgrid(zoomed_xy, zoomed_xy)
                    
                    # Make sure we don't try to sample beyond the array bounds
                    max_idx = min(len(zoomed_xy), zoomed_x.shape[0], zoomed_y.shape[0])
                    if max_idx > 0:
                        # Add quiver with appropriate sampling
                        axs[row_idx, i].quiver(zoomed_xy[:max_idx], zoomed_xy[:max_idx], 
                                              zoomed_x[:max_idx, :max_idx][::zoom_d, ::zoom_d], 
                                              zoomed_y[:max_idx, :max_idx][::zoom_d, ::zoom_d], 
                                              scale=11, color='white', width=0.003)
    
    # Adjust layout to remove extra spacing
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    return fig



def visualize(generator, discriminator, path, dataset):
    
    ##### Define folder paths for training HR and LR data #########
    hr_folder = os.path.join(os.getcwd(), "../../data", dataset, "HR")
    lr_folder = os.path.join(os.getcwd(), "../../data", dataset, "LR")

    ##### Load and combine channels for HR and LR images ########
    hr = load_and_combine_channels(hr_folder)
    lr = load_and_combine_channels(lr_folder)
    # sr_full = generator(lr_images_full)
    # bicubic_full = bicubic_resize(lr_images_full)
    # hr_speed_full = tf.norm(hr_images_full[..., :2], axis=-1)
    # sr_speed_full = tf.norm(sr_full[..., :2], axis=-1)
    # bicubic_speed_full = tf.norm(bicubic_full[..., :2], axis=-1)
    
    # Shuffle for diversity in plots
    batch_size = tf.shape(hr)[0]
    
    # Fixed seed for reproducibility
    tf.random.set_seed(36)
    indices = tf.random.shuffle(tf.range(batch_size))

    hr = tf.gather(hr, indices)
    lr = tf.gather(lr, indices)
    sr = generator(lr)
    bc = bicubic_resize(lr)
    hr_speed = tf.norm(hr[..., :2], axis=-1)
    sr_speed = tf.norm(sr[..., :2], axis=-1)
    bc_speed = tf.norm(bc[..., :2], axis=-1)
    
    ###### Plot 3d fields (vx, vy, ssh) #######
    fig = visualize_batch(lr, hr, sr, bc, fr"$\vec{{V}}$ ({dataset})", field=False)
    fig.savefig(f"{path}/RGB_{dataset}.jpg", dpi=300)

    
    ###### Cast the 2 velocity components into a 3d vector for plotting ######
    lr_vel = lr[..., :2]
    hr_vel = hr[..., :2]
    sr_vel = sr[..., :2]
    bc_vel = bc[..., :2]
    ###### Plot speeds ########
    fig = visualize_batch(lr_vel, hr_vel, sr_vel, bc_vel, fr"Rapidez = $\sqrt{{v_x^2+v_y^2}} ({dataset})$", speed=True)
    
    fig.savefig(f"{path}/V_{dataset}.jpg", dpi=300)
    

    ###### Plot Close-up #######
    bc = bc[..., :2]
    fig = plot_zoom(bc, sr_vel, hr_vel, 3, [[3, 33, 32, 62], [33, 52, 8, 27]], ['red', 'cyan'], field=True)
    plt.show()
    fig.savefig(f'{path}/zoom_{dataset}.jpg', dpi=300)

    ##### Find absolute error #######
    gen_diff = abs(hr_speed - sr_speed)
    bicubic_diff = abs(hr_speed - bc_speed)
    
    image_set = [gen_diff, bicubic_diff]
    
    global_min = min(np.min(images) for images in image_set)
    global_max = max(np.max(images) for images in image_set)

    ##### Plot absolute error #######
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    fig.suptitle(f'Error absoluto de reconstrucción ({dataset})')
    axs[0, 0].set_ylabel('Bicubic')
    axs[1, 0].set_ylabel('SR')
    for i in range(0, 8, 2):
        ax = axs[0, i//2]
        im = ax.imshow(bicubic_diff[i], vmin=global_min, vmax=global_max)
        fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    
        ax = axs[1, i//2]
        im = ax.imshow(gen_diff[i], vmin=global_min, vmax=global_max)
        fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    plt.show()
    fig.savefig(f'{path}/abs_error_{dataset}.jpg', dpi=300)

    ######## Find relative error #########
    gen_diff = abs(hr_speed - sr_speed)/hr_speed
    bicubic_diff = abs(hr_speed - bc_speed)/hr_speed
    
    image_set = [gen_diff, bicubic_diff]
    
    global_min = min(np.min(images) for images in image_set)
    global_max = max(np.max(images) for images in image_set)


    ####### Plot relative error ##########
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    fig.suptitle(f'Error relativo de reconstrucción ({dataset})')
    axs[0, 0].set_ylabel('Bicubic')
    axs[1, 0].set_ylabel('SR')
    for i in range(0, 8, 2):
        ax = axs[0, i//2]
        im = ax.imshow(bicubic_diff[i], vmin=global_min, vmax=2)
        fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    
        ax = axs[1, i//2]
        im = ax.imshow(gen_diff[i], vmin=global_min, vmax=2)
        fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    plt.show()
    fig.savefig(f'{path}/rel_error_{dataset}.jpg', dpi=300)


def plots(path, last=False, mse=False):
    ##### Load model
    generator, discriminator = load_model(path, last, mse)

    # Create a path for figures inside the 'reports' folder.
    splits = path.split("/")
    path = splits[-3] + "/" + splits[-1]
    path = f'E:/Documents/MING/Tesis/Proyecto/reports/figures/{path}'
    
    if last:
        path = f"{path}/last"
    elif mse:
        path = f"{path}/mse"
    else:
        path = f"{path}/best_loss"
        
    os.makedirs(path, exist_ok=True)
    
    print("ON TRAINING DATA")
    visualize(generator, discriminator, path, "train")

    print("\nON VALIDATION DATA")
    visualize(generator, discriminator, path, "val")

    print("\nON TEST DATA")
    visualize(generator, discriminator, path, "test")

def test_(hr, sr, bc):
    ##### 3d "fields" 
    results = evaluate_images(hr, sr, bc)
    means = dict((key, val) for (key, val) in results.items())
    print("\nRGB")
    [print(key, ': ', val) for key, val in means.items()]

    ##### Speeds
    hr_speed = tf.norm(hr[..., :2], axis=-1)
    sr_speed = tf.norm(sr[..., :2], axis=-1)
    bc_speed = tf.norm(bc[..., :2], axis=-1)
    
    results = evaluate_images(hr_speed, sr_speed, bc_speed)
    means = dict((key, val) for (key, val) in results.items())
    print("\nSpeed")
    [print(key, ': ', val) for key, val in means.items()]

def metrics(path, last=False, mse=False):
    
    ##### Load model
    generator, discriminator = load_model(path, last, mse)
    
    ##### Define folder paths for training HR and LR data #########
    
    datasets = [("train", "TRAINING"), ("val", "VALIDATION"), ("test", "TEST")]
    
    for dataset, title in datasets:

        print("\n\nON", title, "DATA")
        
        hr_folder = os.path.join(os.getcwd(), "../../data", dataset, "HR")
        lr_folder = os.path.join(os.getcwd(), "../../data", dataset, "LR")
    
        ##### Load and combine channels for HR and LR images ########
        hr = load_and_combine_channels(hr_folder)
        lr = load_and_combine_channels(lr_folder)
        sr = generator(lr)
        bc = bicubic_resize(lr)

        test_(hr, sr, bc)