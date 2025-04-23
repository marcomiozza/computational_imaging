import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_metrics(gt, reco):
    """
    Computes quality metrics between a reference image and a reconstruction.
    
    Parameters:
    gt : numpy.ndarray (dtype: float32, shape: (H, W))
        Ground truth reference image
        - Values normalized in the range [0, 1]
        - H: image height, W: image width
        
    reco : numpy.ndarray (dtype: float32, shape: (H, W))
        Reconstructed image to evaluate
        - Must have the same dimensions and type as the ground truth
        
    Returns:
    tuple (re, psnr_val, ssim_val) containing:
        re : float
            Relative Error (L2 norm)
            Formula: ||reco - gt||_2 / ||gt||_2
            Range: [0, ∞), where 0 indicates perfect reconstruction
            
        psnr_val : float
            Peak Signal-to-Noise Ratio in decibels (dB)
            Formula: 20 * log10(MAX / sqrt(MSE))
            - MAX=1.0 (data_range specified)
            - Higher values indicate better quality
            Typical range: 20-40 dB for 8-bit images
            
        ssim_val : float
            Structural Similarity Index Measure
            Evaluates structural, luminance, and contrast similarity
            Range: [0, 1], where 1 indicates perfect identity
    """
    # Compute L2 norm of the difference and the ground truth
    # np.linalg.norm computes the Euclidean norm by default (order 2)
    re = np.linalg.norm(reco - gt) / np.linalg.norm(gt)

    # Compute PSNR with maximum range normalized to 1.0
    # psnr automatically handles MSE calculation
    psnr_val = psnr(gt, reco, data_range=1.0)

    # Compute SSIM with dynamic window based on data_range
    # Uses 11x11 Gaussian kernel and sliding window comparison
    ssim_val = ssim(gt, reco, data_range=1.0)  

    return re, psnr_val, ssim_val

def normalize_image(img):
    """
    Normalizes a grayscale image to the range [0, 1].
    
    Parameters:
    img : numpy.ndarray (dtype: any)
        Input image with values in any range
        - Accepts both integer (e.g., uint8, uint16) and float types
        
    Returns:
    numpy.ndarray (dtype: float32)
        Normalized image with float32 values in [0, 1]
        - Special case: uniform array (max=min) returns zeros
        
    Process:
        1. Explicit conversion to float32 for precision
        2. Compute minimum and maximum of the image
        3. Handle division by zero for uniform images
        4. Apply linear formula (img - min)/(max - min)
    """
    # Convert to float32 to avoid overflow/underflow
    img = img.astype(np.float32)

    # Compute statistical extremes
    img_min = np.min(img)  # Minimum pixel value
    img_max = np.max(img)  # Maximum pixel value

    # Handle degenerate case (flat image)
    if img_max == img_min:
        # Return an array of zeros with the same shape/dtype
        return np.zeros_like(img)
        
    # Linear normalization (shift and scaling)
    return (img - img_min) / (img_max - img_min)

def add_noise(sinogram, sigma=0.01):
    """
    Adds additive Gaussian noise to a sinogram.
    
    Parameters:
    sinogram : numpy.ndarray (dtype: any)
        Original sinogram data
        - Format: (num_projections, num_detectors)
        
    sigma : float (optional, default: 0.01)
        Standard deviation of the Gaussian noise
        - Units: same scale as the sinogram
        - Typical values: 0.01-0.1 (1-10% of the signal range)
        
    Returns:
    numpy.ndarray (same dtype as input sinogram)
        Sinogram with additive noise: noisy_sino = sino + N(0, sigma)
        - The dtype is preserved via explicit casting
        
    Process:
        1. Generate Gaussian noise with zero mean
        2. Add to the original sinogram
        3. Convert to the original dtype for compatibility
    """
    # Generate noise with normal distribution
    # Parameters: mean=0, stddev=sigma, shape=sinogram.shape
    noise = np.random.normal(0, sigma, sinogram.shape)

    # Add noise while preserving the original dtype
    # Explicit conversion to avoid altering the data type
    return sinogram + noise.astype(sinogram.dtype)
