import tensorflow as tf
import numpy as np
import os

##############################################################################
# Constants
##############################################################################

R = 6378000.0  # Earth radius in meters
pi = tf.constant(np.pi, dtype=tf.float32)
deg_to_rad = pi / 180.0
g = 9.81 # Gravity
eta = 0 # Viscosity


##############################################################################
# Functions
##############################################################################

##### We need the grid spacing (in meters) of the latice.

def compute_grid_spacing(local=False):
    """
    Computes dx and dy grid spacing in meters from latitude and longitude data. Also computes the
    Coriolis parameter, f.

    Returns:
        dx: Tensor of shape (1, height, width, 1) representing spacing in x direction (meters).
        dy: Tensor of shape (1, height, width, 1) representing spacing in y direction (meters).
        f: Tensor of shape (height, width), representing the value of the Coriolis parameter at each point.
    """

    if local:
        lat = np.load(os.path.join(os.getcwd(), '../../data', 'lat.npy'))
        lng = np.load(os.path.join(os.getcwd(), '../../data', 'lng.npy'))
    else:
        lat = np.load(os.path.join(os.getcwd(), '../data', 'lat.npy'))
        lng = np.load(os.path.join(os.getcwd(), '../data', 'lng.npy'))
    
    #
    lat = lat[::-1]
    
    # Create 2D grids using meshgrid
    lat_grid, lng_grid = np.meshgrid(lat, lng, indexing="ij")
    
    # Convert to tensors
    lat = tf.convert_to_tensor(lat_grid, dtype=tf.float32)  # shape (height, width)
    lng = tf.convert_to_tensor(lng_grid, dtype=tf.float32)  # shape (height, width)

    f = 2 * (2 * pi / 86400) * tf.sin(lat_grid * deg_to_rad)
    
    # Compute dy (North-South distance)
    lat_diff = lat[1:, :] - lat[:-1, :]  # Difference in latitude
    dy = R * deg_to_rad * lat_diff  # Convert to meters

    # Compute dx (East-West distance), considering latitude
    lng_diff = lng[:, 1:] - lng[:, :-1]  # Difference in longitude
    dx = R * tf.cos(lat[:, :-1] * deg_to_rad) * deg_to_rad * lng_diff  # Convert to meters

    # Pad on the corresponding axis by repeating the first and last row/column
    dx_left = tf.concat([dx[:, :1], dx[:, :-1]], axis=1)
    dx_right = tf.concat([dx[:, 1:], dx[:, -1:]], axis=1)
    dx = 0.5 * (dx_left + dx_right)
    dx = tf.concat([dx[:, :1], dx], axis=1)
    
    dy_top = tf.concat([dy[:1, :], dy[:-1, :]], axis=0)
    dy_bottom = tf.concat([dy[1:, :], dy[-1:, :]], axis=0)
    dy = 0.5 * (dy_top + dy_bottom)
    dy = tf.concat([dy[:1, :], dy], axis=0)

    # Reshape for use in derivation functions.
    
    dx = tf.reshape(dx, [1, dx.shape[0], dx.shape[1], 1])
    dy = tf.reshape(dy, [1, dy.shape[0], dy.shape[1], 1])
    
    return dx, dy, f


def gaussian_kernel1d(size, sigma, order=0):
    """
    Creates a 1D Gaussian kernel or its 1st or 2nd order derivative.

    Args:
        - size (int): Kernel length (odd number).
        - sigma (float): Standard deviation of the Gaussian.
        - order (int): Derivative order.

    Returns:
        tf.Tensor: Normalized 1D kernel tensor to be used in convolution.
    """
    if order < 0:
        raise ValueError("order must be non-negative")

    x = np.arange(-size // 2 + 1, size // 2 + 1)
    sigma2 = sigma ** 2

    if order == 0:
        kernel = np.exp(-0.5 * (x ** 2) / sigma2)
        kernel /= kernel.sum()
    elif order == 1:
        kernel = -x * np.exp(-0.5 * (x ** 2) / sigma2)
        kernel /= np.sum(x * kernel)
    elif order == 2:
        kernel = (x ** 2 - sigma2) * np.exp(-0.5 * (x ** 2) / sigma2)
        kernel /= np.sum(0.5 * (x ** 2) * kernel)
    else:
        raise NotImplementedError("Higher-order derivatives not implemented")

    return tf.convert_to_tensor(kernel, dtype=tf.float32)

def gradient(image, size=5, sigma=1.0, dx=1.0, dy=1.0):
    """
    Computes the gradient of a single-channel 2D image using independent Gaussian derivative kernels for the x and y directions.

    Args:
        - image (tf.Tensor or array-like): Input image (H x W), or batch (B x H x W), single channel.
        - size (int): Kernel size (should be odd).
        - sigma (float): Standard deviation for the Gaussian kernel.
        - dx: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in x (longitude).
        - dy: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in y (latitude).
    Returns:
        tf.Tensor: Gradient tensor with shape (B, H, W, 2), containing gradients along x and y.
    """
    image = tf.convert_to_tensor(image)

    if len(image.shape) == 2:
        image = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)
    elif len(image.shape) == 3:
        if image.shape[-1] != 1:  # sospechoso, podría ser (B, H, W) sin canal
            image = tf.expand_dims(image, axis=-1)
    elif len(image.shape) == 4 and image.shape[-1] != 1:
        raise ValueError(f"Expected single-channel input, got shape {image.shape}")

    pad = size // 2

    # Separate padding for each direction
    image_pad_x = tf.pad(image, [[0, 0], [0, 0], [pad, pad], [0, 0]], mode='SYMMETRIC')  # pad width
    image_pad_y = tf.pad(image, [[0, 0], [pad, pad], [0, 0], [0, 0]], mode='SYMMETRIC')  # pad height

    kernel_x = gaussian_kernel1d(size, sigma, 1)
    kernel_y = gaussian_kernel1d(size, sigma, 1)

    kernel_x = tf.reshape(kernel_x, [1, size, 1, 1])  # horizontal
    kernel_y = tf.reshape(kernel_y, [size, 1, 1, 1])  # vertical

    grad_x = tf.nn.conv2d(image_pad_x, kernel_x, strides=[1, 1, 1, 1], padding="VALID")
    grad_y = tf.nn.conv2d(image_pad_y, kernel_y, strides=[1, 1, 1, 1], padding="VALID")
    
    grad_x = grad_x/dx
    grad_y = grad_y/dy
    
    return tf.concat([grad_x, grad_y], axis=-1)



@tf.function
def advection(sr, dx=1.0, dy=1.0):
    """
    Computes the advective term (u · ∇)u using Gaussian filters derivatives.

    Args:
    - sr: tf.Tensor of shape (batch, height, width, 3), where:
          u[..., 0] = u_x, u[..., 1] = u_y, u[..., 2] = SSH
    - dx: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in x (longitude).
    - dy: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in y (latitude).

    Returns:
    - adv: tf.Tensor of shape (batch, height, width, 2), containing ((u · ∇)u, (v · ∇)u).
    """

    # Compute velocity gradients
    du_grad = gradient(sr[..., 0], dx=dx, dy=dy) 
    du_dx = du_grad[..., 0]
    du_dy = du_grad[..., 1]
    
    dv_grad = gradient(sr[..., 1], dx=dx, dy=dy)
    dv_dx = dv_grad[..., 0]
    dv_dy = dv_grad[..., 1]
    
    # Compute advective terms
    adv_u = sr[..., 0] * du_dx + sr[..., 1] * du_dy
    adv_v = sr[..., 0] * dv_dx + sr[..., 1] * dv_dy

    adv = tf.concat([adv_u, adv_v], axis=-1)  # Shape: (batch, height, width, 2)
    return adv

@tf.function  # Ensures compatibility with Graph Mode
def laplacian(field, size=5, sigma=1.0, dx=1.0, dy=1.0):
    """
    Computes the Laplacian of a 2D vector field using Gaussian second-derivative kernels.

    Args:
        - field (tf.Tensor): Input tensor of shape (batch, height, width, 2) or (height, width, 2). The last dimension corresponds to vector components (u, v).
        - size (int): Kernel size (odd integer).
        - sigma (float): Standard deviation for the Gaussian kernel.
        - dx: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in x (longitude).
        - dy: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in y (latitude).

    Returns:
        tf.Tensor: Laplacian tensor with shape (batch, height, width, 2), containing Laplacians of u and v.
    """
    if len(field.shape) == 3:  # (H, W, 2)
        field = tf.expand_dims(field, axis=0)  # (1, H, W, 2)

    u = field[..., 0]  # (B, H, W, 1)
    v = field[..., 1]  # (B, H, W, 1)

    def laplacian_scalar(f):
        pad = size // 2

        f_pad_xx = tf.pad(f, [[0, 0], [0, 0], [pad, pad], [0, 0]], mode='SYMMETRIC')
        f_pad_yy = tf.pad(f, [[0, 0], [pad, pad], [0, 0], [0, 0]], mode='SYMMETRIC')

        kernel_xx = gaussian_kernel1d(size, sigma, order=2)
        kernel_yy = gaussian_kernel1d(size, sigma, order=2)

        kernel_xx = tf.reshape(kernel_xx, [1, size, 1, 1])
        kernel_yy = tf.reshape(kernel_yy, [size, 1, 1, 1])

        lap_xx = tf.nn.conv2d(f_pad_xx, kernel_xx, strides=[1, 1, 1, 1], padding="VALID") / (dx ** 2)
        lap_yy = tf.nn.conv2d(f_pad_yy, kernel_yy, strides=[1, 1, 1, 1], padding="VALID") / (dy ** 2)

        return lap_xx + lap_yy

    lap_u = laplacian_scalar(u)
    lap_v = laplacian_scalar(v)

    return tf.concat([lap_u, lap_v], axis=-1)  # (B, H, W, 2)

@tf.function
def navier_stokes_loss(sr, f, laplace=False, adv=False, dx=1.0, dy=1.0, means=None, stds=None):
    """
    Computes the mean L2 norm of the residuals from the shallow water Navier-Stokes equations.

    This loss function evaluates the physical consistency of a predicted state `sr` by computing
    the residuals of the momentum equations, including optional viscous diffusion (Laplacian)
    and advection terms.

    Args:
        - sr (tf.Tensor): Tensor representing a super-resolved field of shape (batch, height, width, 3), 
          where:
                    - sr[..., 0] is u (zonal velocity),
                    - sr[..., 1] is v (meridional velocity),
                    - sr[..., 2] is SSH (sea surface height).
        - laplace (bool): If True, includes the viscous diffusion term using the Laplacian of velocity.
        - adv (bool): If True, includes the nonlinear advection term (u · ∇)u.
        - dx: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in x (longitude).
        - dy: Scalar or tf.Tensor of shape (1, height, width, 1), grid spacing in y (latitude).
        - means: dictionary with mean values for 'vx', 'vy' and 'ssh' (as tensors or floats)
        - stds: dictionary with std values for 'vx', 'vy' and 'ssh' (as tensors or floats)
    Returns:
        tf.Tensor: Scalar tensor representing the mean L2 norm of the residuals across the domain.
    """
    if means is None or stds is None:
        raise ValueError("Both `means` and `stds` must be provided.")

    # De-standardize u, v, ssh
    u = sr[..., 0] * stds['vx'] + means['vx']
    v = sr[..., 1] * stds['vy'] + means['vy']
    ssh = sr[..., 2] * stds['ssh'] + means['ssh']

    grad_ssh = gradient(ssh, dx=dx, dy=dy)
    coriolis = tf.stack([-f * v, f * u], axis=-1)
    nav_stokes = g * grad_ssh + coriolis

    if laplace:
        nav_stokes -= eta * laplacian(tf.stack([u, v], axis=-1), dx=dx, dy=dy)

    if adv:
        full_field = tf.stack([u, v, ssh], axis=-1)
        nav_stokes += advection(full_field, dx=dx, dy=dy)

    return tf.reduce_mean(tf.norm(nav_stokes, axis=-1, ord=2))


@tf.function
def divergence_loss(sr, dx, dy, means=None, stds=None):
    if means is None or stds is None:
        raise ValueError("Both `means` and `stds` must be provided.")

    u = sr[..., 0] * stds['vx'] + means['vx']
    v = sr[..., 1] * stds['vy'] + means['vy']

    grad_u = gradient(u, dx=dx, dy=dy) # (du/dx, du/dy)
    grad_v = gradient(v, dx=dx, dy=dy) # (dv/dx, dv/dy)

    divergence = grad_u[..., 0] + grad_v[..., 1]

    return tf.reduce_mean(tf.norm(divergence, ord=1, axis=[1, 2]))