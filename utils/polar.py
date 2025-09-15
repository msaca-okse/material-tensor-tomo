import numpy as np



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