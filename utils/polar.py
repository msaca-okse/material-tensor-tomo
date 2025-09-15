import numpy as np
import cupy as cp



def cartesian_to_polar(matrix, num_phi=360, num_rad=None, max_radius = None,factor = 3):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    rows, cols = matrix.shape
    if max_radius is None:
        max_radius = (min(cols, rows) // 2)
    if num_rad is None:
        num_rad = max_radius


    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 0, 0
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    

    # Create polar coordinate grid
    theta = np.linspace(0, 2 * np.pi, num_phi)  # Angles
    r = np.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = np.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * np.cos(Theta)
    Y = center_y + R * np.sin(Theta)

    # Bilinear interpolation
    X = np.clip(X, 0, cols - 1)
    Y = np.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(np.int32), Y.astype(np.int32)  # Floor values
    x1, y1 = np.clip(x0 + 1, 0, cols - 1), np.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix, max_radius





def cartesian_to_polar_cupy(matrix, num_phi=360, num_rad=None, max_radius = None,factor = 3):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    rows, cols = matrix.shape
    if max_radius is None:
        max_radius = (min(cols, rows) // 2)
    if num_rad is None:
        num_rad = max_radius


    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 0, 0
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    

    # Create polar coordinate grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)  # Angles
    r = cp.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = cp.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # Bilinear interpolation
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)  # Floor values
    x1, y1 = cp.clip(x0 + 1, 0, cols - 1), cp.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix, max_radius




def detector_radius_to_twotheta(
    det_array,
    two_theta_new,
    detector_distance,
    r_max
):
    """
    Interpolate detector data from radius -> 2θ space.

    Parameters
    ----------
    det_array : ndarray, shape (n_azimuth, n_radius)
        Input array, axis=0 azimuth, axis=1 radius.
    two_theta_new : 1D ndarray
        Target 2θ values in degrees. Must lie within detector range.
    detector_distance : float
        Sample-to-detector distance (same units as r_max).
    r_max : float
        Maximum detector radius (same units as detector_distance).

    Returns
    -------
    out_array : ndarray, shape (n_azimuth, len(two_theta_new))
        Rebinned array in azimuth × 2θ space.
    """
    n_azimuth, n_radius = det_array.shape

    # radius axis (0..r_max)
    r = np.linspace(0, r_max, n_radius)

    # convert radius -> 2θ (degrees)
    two_theta = np.degrees(np.arctan(r / detector_distance))

    # range check
    if (two_theta_new.min() < two_theta.min()) or (two_theta_new.max() > two_theta.max()):
        raise ValueError(
            f"Requested 2θ range {two_theta_new.min()}–{two_theta_new.max()} deg "
            f"outside detector range {two_theta.min()}–{two_theta.max()} deg."
        )

    # interpolate along axis=1 (radius → 2θ)
    out = np.empty((n_azimuth, len(two_theta_new)), dtype=det_array.dtype)
    for i in range(n_azimuth):
        out[i, :] = np.interp(two_theta_new, two_theta, det_array[i, :])

    return out




# def detector_radius_to_twotheta_cupy(
#     det_array: cp.ndarray,
#     two_theta_new,
#     detector_distance: float,
#     r_max: float
# ):
#     """
#     Interpolate detector data from radius -> 2θ space (CuPy version).
#     """
#     n_azimuth, n_radius = det_array.shape

#     # make sure inputs are cupy
#     two_theta_new = cp.asarray(two_theta_new)

#     # radius axis
#     r = cp.linspace(0, r_max, n_radius)

#     # convert radius -> 2θ (degrees)
#     two_theta = cp.degrees(cp.arctan(r / detector_distance))

#     # range check (cast to float to avoid cp.bool_ -> error)
#     if (float(two_theta_new.min()) < float(two_theta.min())) or \
#        (float(two_theta_new.max()) > float(two_theta.max())):
#         raise ValueError("Requested 2θ range outside detector range.")

#     # interpolate along axis=1
#     out = cp.empty((n_azimuth, len(two_theta_new)), dtype=det_array.dtype)
#     for i in range(n_azimuth):
#         out[i, :] = cp.interp(two_theta_new, two_theta, det_array[i, :])

#     return out


def detector_radius_to_twotheta_cupy(det_array, two_theta_new, detector_distance, r_max):
    n_azimuth, n_radius = det_array.shape
    two_theta_new = cp.asarray(two_theta_new)

    r = cp.linspace(0, r_max, n_radius)
    two_theta = cp.degrees(cp.arctan(r / detector_distance))

    # indices of bins
    idx = cp.searchsorted(two_theta, two_theta_new, side="left")
    idx = cp.clip(idx, 1, n_radius-1)  # valid range

    # x0, x1
    x0 = two_theta[idx-1]
    x1 = two_theta[idx]

    # values y0, y1 for all azimuths
    y0 = det_array[:, idx-1]   # shape (n_azimuth, n_new)
    y1 = det_array[:, idx]

    # linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    out = y0 + slope * (two_theta_new - x0)

    return out




import cupy as cp

def cartesian_to_polar_cupy_batch(matrix, num_phi=360, num_rad=None, max_radius=None, factor=3):
    """
    Batched Cartesian → Polar for CuPy arrays.

    Parameters
    ----------
    matrix : cp.ndarray, shape (B, H, W)
        Batch of 2D images.
    num_phi : int
        Number of angular bins.
    num_rad : int or None
        Number of radial bins (default = min(H,W)//2).
    max_radius : int or None
        Maximum radius (default = min(H,W)//2).
    factor : int
        Oversampling factor (for antialiasing).

    Returns
    -------
    polar_matrix : cp.ndarray, shape (B, num_phi, num_rad)
    max_radius : int
    """
    B, rows, cols = matrix.shape
    if max_radius is None:
        max_radius = min(cols, rows) // 2
    if num_rad is None:
        num_rad = max_radius

    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor * num_phi
    num_rad = factor * num_rad

    center_x, center_y = cols // 2, rows // 2

    # polar grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)
    r = cp.linspace(0, max_radius, num_rad)
    R, Theta = cp.meshgrid(r, theta)   # (num_phi, num_rad)

    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # clip
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, cols - 1)
    y1 = cp.clip(y0 + 1, 0, rows - 1)

    # expand to batch
    x0 = cp.broadcast_to(x0[None, :, :], (B, num_phi, num_rad))
    x1 = cp.broadcast_to(x1[None, :, :], (B, num_phi, num_rad))
    y0 = cp.broadcast_to(y0[None, :, :], (B, num_phi, num_rad))
    y1 = cp.broadcast_to(y1[None, :, :], (B, num_phi, num_rad))

    # bilinear interpolation
    Ia = matrix[cp.arange(B)[:, None, None], y0, x0]
    Ib = matrix[cp.arange(B)[:, None, None], y0, x1]
    Ic = matrix[cp.arange(B)[:, None, None], y1, x0]
    Id = matrix[cp.arange(B)[:, None, None], y1, x1]

    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    polar_matrix = wa*Ia + wb*Ib + wc*Ic + wd*Id
    polar_matrix = polar_matrix.reshape(B, num_phi_old, factor, num_rad_old, factor)
    polar_matrix = polar_matrix.mean(axis=(2,4))

    return polar_matrix, max_radius



def detector_radius_to_twotheta_cupy_batch(det_array, two_theta_new, detector_distance, r_max):
    """
    det_array: (B, n_azimuth, n_radius)
    two_theta_new: (n_new,)
    """
    B, n_azimuth, n_radius = det_array.shape
    two_theta_new = cp.asarray(two_theta_new)

    r = cp.linspace(0, r_max, n_radius)
    two_theta = cp.degrees(cp.arctan(r / detector_distance))

    idx = cp.searchsorted(two_theta, two_theta_new, side="left")
    idx = cp.clip(idx, 1, n_radius-1)

    x0, x1 = two_theta[idx-1], two_theta[idx]
    y0 = det_array[:, :, idx-1]   # (B, n_azimuth, n_new)
    y1 = det_array[:, :, idx]

    slope = (y1 - y0) / (x1 - x0)
    out = y0 + slope * (two_theta_new - x0)

    return out   # shape (B, n_azimuth, n_new)


import cupy as cp

def process_diffraction_cupy(
    diffraction_4d_gpu, 
    num_phi, factor, 
    two_theta_new, detector_distance, r_max, 
    chunk_size=64
):
    """
    Process diffraction data (Cartesian->Polar->2θ) in chunks on GPU.

    Parameters
    ----------
    diffraction_4d_gpu : cp.ndarray, shape (Nx, Ny, H, W)
        Input diffraction data on GPU.
    num_phi : int
        Number of azimuth bins.
    factor : int
        Oversampling factor for antialiasing in polar conversion.
    two_theta_new : 1D array-like
        Target 2θ values in degrees.
    detector_distance : float
        Sample-to-detector distance.
    r_max : float
        Maximum detector radius.
    chunk_size : int
        Number of slices per chunk.

    Returns
    -------
    out : cp.ndarray, shape (Nx, Ny, num_phi, len(two_theta_new))
    """
    Nx, Ny, H, W = diffraction_4d_gpu.shape
    B = Nx * Ny
    two_theta_new_gpu = cp.asarray(two_theta_new)

    # allocate output array on GPU
    out = None

    # flatten (Nx,Ny) into batch dimension
    batch = diffraction_4d_gpu.reshape(B, H, W)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = batch[start:end]  # shape (b, H, W)

        # Cartesian -> Polar
        pol_chunk, max_rad = cartesian_to_polar_cupy_batch(
            chunk, num_phi=num_phi, factor=factor
        )  # (b, num_phi, num_rad)

        # radius -> 2θ
        pol_reb_chunk = detector_radius_to_twotheta_cupy_batch(
            pol_chunk, two_theta_new_gpu, detector_distance, r_max
        )  # (b, num_phi, n_new)

        # allocate output once
        if out is None:
            out = cp.empty(
                (B, pol_reb_chunk.shape[1], pol_reb_chunk.shape[2]),
                dtype=pol_reb_chunk.dtype,
            )

        out[start:end] = pol_reb_chunk

    # reshape back to (Nx, Ny, num_phi, n_new)
    out = out.reshape(Nx, Ny, out.shape[1], out.shape[2])
    return out
