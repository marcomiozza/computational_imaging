import numpy as np
import astra

def gradient(u):
    """
    Computes the discrete gradient of a 2D image using forward finite differences.
    
    Parameters:
    u : numpy.ndarray, shape (M, N)
        Input grayscale image
        - M: number of rows (y-axis)
        - N: number of columns (x-axis)
    
    Returns:
    tuple (gx, gy) where:
        gx : numpy.ndarray, shape (M, N)
            Horizontal gradient (differences along the x-axis)
        gy : numpy.ndarray, shape (M, N)
            Vertical gradient (differences along the y-axis)
    """
    gx = np.roll(u, -1, axis=1) - u  # Right-left difference
    # np.roll(array, shift=-1, axis=1): shifts elements 1 position left along columns (x)

    gy = np.roll(u, -1, axis=0) - u  # Bottom-top difference
    # np.roll(array, shift=-1, axis=0): shifts elements 1 position up along rows (y)
    
    # Edge correction to avoid circularity
    gx[:, -1] = 0  # Last column: cancel roll effect (right edge)
    gy[-1, :] = 0  # Last row: cancel roll effect (bottom edge)
    
    return gx, gy

def divergence(px, py):
    """
    Computes the discrete divergence (adjoint of the gradient) for a 2D vector field.
    
    Parameters:
    px : numpy.ndarray, shape (M, N)
        Horizontal component of the vector field
    py : numpy.ndarray, shape (M, N)
        Vertical component of the vector field
    
    Returns:
    numpy.ndarray, shape (M, N)
        Scalar field resulting from the divergence
    """
    fx = px - np.roll(px, 1, axis=1)  # Left-right difference
    # np.roll(..., 1, axis=1): shifts elements 1 position right (x-axis)

    fy = py - np.roll(py, 1, axis=0)  # Top-bottom difference
    # np.roll(..., 1, axis=0): shifts elements 1 position down (y-axis)
    
    # Edge correction to avoid circular propagation
    fx[:, 0] = px[:, 0]  # First column: original value (left edge)
    fy[0, :] = py[0, :]  # First row: original value (top edge)
    
    return fx + fy

def chambolle_pock_tv_reconstruction(sinogram, angles_deg, lambda_tv=0.1, iterations=300):
    """
    Tomographic reconstruction with Total Variation regularization using the Chambolle-Pock algorithm.
    
    Parameters:
    sinogram : numpy.ndarray, shape (num_angles, det_count)
        Sinogram data in the format:
        - num_angles: number of projection angles
        - det_count: number of detectors per projection
        
    angles_deg : array-like, shape (num_angles,)
        Acquisition angles in degrees
        Example: np.linspace(0, 180, 180, endpoint=False)
        
    lambda_tv : float, optional (default: 0.1)
        Total Variation regularization parameter
        - Higher values: more smoothing, less detail
        - Lower values: less smoothing, more noise
        
    iterations : int, optional (default: 300)
        Total number of iterations of the algorithm
        - Typically between 100-1000 depending on complexity
        
    Returns:
    numpy.ndarray, shape (vol_size, vol_size)
        Reconstructed image with values in [0,1]
    """
    # Convert angles from degrees to radians (required by ASTRA)
    angles_rad = np.deg2rad(angles_deg)
    
    # Number of detectors (determines spatial resolution)
    det_count = sinogram.shape[1]
    
    # Reconstructed volume size (assumed square)
    vol_size = det_count  # Common convention in tomography

    # Create projection geometry for ASTRA
    # Parameters of create_proj_geom:
    # 'parallel': type of geometry (parallel rays)
    # 1.0: spacing between detectors (arbitrary units)
    # det_count: number of detector elements
    # angles_rad: array of angles in radians
    proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles_rad)
    
    # Create volume geometry for ASTRA
    # Parameters of create_vol_geom:
    # vol_size: x-dimension (columns)
    # vol_size: y-dimension (rows)
    vol_geom = astra.create_vol_geom(vol_size, vol_size)

    # Create ASTRA projector (core of geometric operations)
    # 'linear' indicates linear interpolation (bilinear 2D)
    # Projector links projection and volume geometries
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)

    # Define forward projection operator (Radon transform)
    def A(x):
        """Performs forward projection (discrete Radon transform)."""
        # Create ASTRA volume data structure
        # '-vol': volume data type
        # vol_geom: previously defined geometry
        # data=x: volume density values
        id_vol = astra.data2d.create('-vol', vol_geom, data=x)
        
        # Create empty sinogram data structure
        id_proj = astra.data2d.create('-sino', proj_geom)
        
        # Configure Forward Projection (FP) algorithm
        cfg = astra.astra_dict('FP')  # FP = Forward Projection
        cfg['ProjectorId'] = projector_id  # Use created projector
        cfg['VolumeDataId'] = id_vol  # Input: volume
        cfg['ProjectionDataId'] = id_proj  # Output: sinogram
        cfg['ReconstructionDataId'] = id_vol  # Not used in FP
        
        # Execute algorithm
        id_alg = astra.algorithm.create(cfg)
        astra.algorithm.run(id_alg)  # Perform the projection
        
        # Extract data from ASTRA to numpy array
        y = astra.data2d.get(id_proj)
        
        # Clean up ASTRA memory (critical to avoid memory leaks)
        astra.algorithm.delete(id_alg)
        astra.data2d.delete(id_proj)
        astra.data2d.delete(id_vol)
        
        return y

    # Define adjoint operator (backprojection)
    def AT(y):
        """Performs backprojection (adjoint of the Radon transform)."""
        # Create ASTRA sinogram data structure
        id_proj = astra.data2d.create('-sino', proj_geom, data=y)
        
        # Create empty volume data structure
        id_vol = astra.data2d.create('-vol', vol_geom)
        
        # Configure Back Projection (BP) algorithm
        cfg = astra.astra_dict('BP')  # BP = Back Projection
        cfg['ProjectorId'] = projector_id
        cfg['ProjectionDataId'] = id_proj  # Input: sinogram
        cfg['ReconstructionDataId'] = id_vol  # Output: volume
        
        # Execute algorithm
        id_alg = astra.algorithm.create(cfg)
        astra.algorithm.run(id_alg)
        
        # Extract data
        x = astra.data2d.get(id_vol)
        
        # Clean up memory
        astra.algorithm.delete(id_alg)
        astra.data2d.delete(id_proj)
        astra.data2d.delete(id_vol)
        
        return x

    # Initialize algorithm variables
    x = np.zeros((vol_size, vol_size), dtype=np.float32)  # Primal image
    px = np.zeros_like(x)  # Dual variable for x-component of gradient
    py = np.zeros_like(x)  # Dual variable for y-component of gradient
    y = np.zeros_like(sinogram)  # Dual variable for data fidelity

    # Algorithm parameters (Chambolle-Pock optimization)
    tau = 0.01  # Update step for primal variable (x)
    sigma = 0.01  # Update step for dual variables (y, px, py)
    theta = 1.0  # Extrapolation factor for accelerated convergence
    x_bar = x.copy()  # Auxiliary variable for extrapolation

    b = sinogram  # Observed sinogram (measured data)

    # Main optimization loop
    for k in range(iterations):
        # Update dual variable y (data fidelity term)
        grad_x_bar = A(x_bar)  # Compute Ax (forward projection)
        y += sigma * (grad_x_bar - b)  # Update residual
        y /= (1.0 + sigma)  # Normalize for numerical stability

        # Update dual variables px, py (TV regularization)
        gx, gy = gradient(x_bar)  # Compute gradient of x_bar
        px += sigma * gx  # Update x-component of dual
        py += sigma * gy  # Update y-component of dual
        
        # Projection onto dual ball (enforcing ||(px,py)||_2 <= lambda_tv)
        norm = np.maximum(1.0, np.sqrt(px**2 + py**2) / lambda_tv)
        px /= norm  # Normalize px
        py /= norm  # Normalize py

        # Update primal variable x
        x_old = x.copy()  # Save previous state for extrapolation
        x -= tau * (AT(y) + divergence(px, py))  # Gradient descent step
        x = np.clip(x, 0, 1)  # Physical constraint (values between 0 and 1)

        # Compute x_bar for next iteration (Nesterov acceleration)
        x_bar = x + theta * (x - x_old)  # Extrapolation step

    # Final cleanup of ASTRA projector (free GPU/CPU memory)
    astra.projector.delete(projector_id)
    
    return x  # Returns the reconstructed normalized image
