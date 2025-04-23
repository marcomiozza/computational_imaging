import numpy as np  
import astra    
import os   
from skimage.io import imread   
from skimage.transform import resize  
from src.utils import add_noise, normalize_image  

def simulate_sinogram(image, angles_deg, add_noise_flag=False, noise_sigma=0.01):
    """
    Simulates the acquisition of a sinogram from a 2D image.
    
    Parameters:
    image : numpy.ndarray 
        Original grayscale image
        - Format: (H, W) with arbitrary values
        
    angles_deg : array-like 
        Projection angles in degrees
        - Example: np.linspace(0, 180, 180) for 180 projections
        
    add_noise_flag : bool (optional, default: False)
        Enables the addition of Gaussian noise
        
    noise_sigma : float (optional, default: 0.01)
        Standard deviation of the noise (only if add_noise_flag=True)
        
    Returns:
    tuple (sinogram, proj_geom, vol_geom) containing:
        sinogram : numpy.ndarray 
            Simulated sinogram (shape: num_angles x num_detectors)
        proj_geom : astra geometry object
            Projection geometry used
        vol_geom : astra geometry object
            Volume geometry used
    """
    # Image preprocessing
    image = normalize_image(image)  # Normalize to [0,1]
    # Ensures values are in [0,1] regardless of the original input

    image = resize(image, (512, 512), anti_aliasing=True)  # Resize to 512x512
    # Standardizes the size for compatibility with tomographic algorithms
    
    # ASTRA geometry configuration
    angles_rad = np.deg2rad(angles_deg)  # Convert angles to radians
    
    vol_geom = astra.create_vol_geom(image.shape[0], image.shape[1])  # Create 512x512 volume
    # Defines a square geometry based on the image dimensions
    
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[0], angles_rad)  # Parallel geometry
    # 'parallel': Type of geometry (parallel rays)
    # 1.0: Detector spacing (arbitrary ASTRA units)
    # image.shape[0]: Number of detectors = image size
    
    # ASTRA projector initialization
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)  # Linear projector
    
    # ASTRA data structures creation
    sinogram_id = astra.data2d.create('-sino', proj_geom)  # Empty sinogram
    image_id = astra.data2d.create('-vol', vol_geom, data=image)  # Volume from image
    
    # Forward Projection algorithm configuration
    cfg = astra.astra_dict('FP')  # Forward Projection
    cfg['ProjectorId'] = proj_id  # Link projector
    cfg['ProjectionDataId'] = sinogram_id  # Sinogram destination
    cfg['VolumeDataId'] = image_id  # Volume source
    
    # Algorithm execution
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)  # Generate sinogram
    
    # Retrieve data from ASTRA
    sinogram = astra.data2d.get(sinogram_id)  # Convert to numpy array
    
    # Conditional noise addition
    if add_noise_flag:
        sinogram += add_noise(sinogram, sigma=noise_sigma)  # Add Gaussian noise
        # Adds noise while preserving the original dtype
    
    # Clean up ASTRA resources
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(image_id)
    astra.projector.delete(proj_id)
    
    return sinogram, proj_geom, vol_geom

def generate_and_save_sinograms(image_path, output_folder, noise_sigma=0.01):
    """
    Generates and saves multiple variants of sinograms for an image.
    
    Parameters:
    image_path : str 
        Path to the original image file
        - Supported formats: PNG, JPG, TIFF, etc.
        
    output_folder : str 
        Output directory to save the sinograms
        - Will be created if it does not exist
        
    noise_sigma : float (optional, default: 0.01)
        Noise intensity for noisy variants
        
    Generated configurations:
        - limited_clean: 30 projections (-30° to 30°), no noise
        - limited_noisy: 30 projections with noise
        - full_clean: 180 projections (-90° to 90°), no noise
        - full_noisy: 180 projections with noise
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)  # Recursively create if necessary
    # os.makedirs(...): Creates the full directory hierarchy
    
    # Load image
    image = imread(image_path, as_gray=True)  # Load as grayscale
    # imread(..., as_gray=True): Forces conversion to grayscale even for RGB images
    
    # Acquisition configurations
    configs = [
        ("limited_clean", np.linspace(-30, 30, 30), False),  # Clean limited scan
        # np.linspace(-30,30,30): 30 equally spaced angles between -30° and 30°

        ("limited_noisy", np.linspace(-30, 30, 30), True),   # Noisy limited scan
        ("full_clean", np.linspace(-90, 90, 180), False),    # Clean full scan
        # np.linspace(-90,90,180): 180 angles for full coverage (1° angular step)

        ("full_noisy", np.linspace(-90, 90, 180), True),     # Noisy full scan
    ]
    
    # Generate sinograms
    for name, angles, noise in configs:
        # Simulate sinogram
        sinogram, _, _ = simulate_sinogram(
            image, 
            angles, 
            add_noise_flag=noise, 
            noise_sigma=noise_sigma
        ) 
        
        # Simulated sinogram with specified angles and noise 
        
        # Save as .npy file
        np.save(
            os.path.join(output_folder, f"{name}_sinogram.npy"), 
            sinogram
        )
        # np.save(...): Saves in numpy binary format to preserve precision
