import numpy as np
import os
from src.utils import compute_metrics, normalize_image
from skimage.io import imread, imsave
import pandas as pd
from src.total_variation import chambolle_pock_tv_reconstruction

def evaluate_single_reconstruction(
    gt_path, sinogram_path, angles,
    method='tv', lambda_tv=0.1, iterations=300,
    save_path=None, verbose=False
):
    """
    Reconstructs a single image from a sinogram and compares it to the ground truth.

    Parameters:
    - gt_path (str): Path to the ground truth image file.
    - sinogram_path (str): Path to the corresponding sinogram (NumPy .npy file).
    - angles (np.ndarray): Array of projection angles in radians.
    - method (str): Reconstruction method, default is 'tv' (Total Variation).
    - lambda_tv (float): Regularization parameter for TV.
    - iterations (int): Number of optimization iterations.
    - save_path (str or None): Path where to save the reconstructed image (if not None).
    - verbose (bool): If True, prints the result of this reconstruction.

    Returns:
    - dict: A dictionary with metrics and identifiers.
    """
    # Load and normalize the ground truth image
    gt = imread(gt_path, as_gray=True)
    gt = normalize_image(gt)

    # Load the sinogram from file
    sinogram = np.load(sinogram_path)

    # Perform reconstruction using the chosen method
    if method == 'tv':
        reco = chambolle_pock_tv_reconstruction(sinogram, angles, lambda_tv=lambda_tv, iterations=iterations)
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")

    # Ensure same shape between ground truth and reconstruction
    gt = np.resize(gt, reco.shape)

    # Compute RE, PSNR and SSIM
    re, psnr_val, ssim_val = compute_metrics(gt, reco)

    # Save the image if requested
    if save_path:
        imsave(save_path, (reco * 255).astype(np.uint8))

    # Optional logging
    if verbose:
        print(f"  - GT: {os.path.basename(gt_path)} | Lambda: {lambda_tv:.4f} | RE={re:.4f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")

    # Return results in a dictionary
    return {
        "GT": os.path.basename(gt_path),
        "Sinogram": os.path.basename(sinogram_path),
        "Lambda": lambda_tv,
        "Iterations": iterations,
        "RE": re,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "Min": reco.min(),
        "Max": reco.max()
    }
    
    
    # Min: The minimum pixel value in the reconstructed image.
    # Max: The maximum pixel value in the reconstructed image.

def evaluate_batch(config):
    """
    Runs a full experiment batch using the parameters defined in a configuration dictionary.

    Parameters:
    - config (dict): Dictionary with keys like:
        - "image_folder" or "images"
        - "tests" with sinogram and angle info
        - "lambda_values", "iterations"
        - "save_images", "save_only_first_image"
        - "reconstruction_folder"

    Returns:
    - list: List of dictionaries with results for each reconstruction.
    """
    results = []

    # Load all images from folder or predefined list
    if "image_folder" in config:
        images = [os.path.join(config["image_folder"], f)
                  for f in os.listdir(config["image_folder"])
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        images = [img["gt_path"] for img in config["images"]]

    # Check if we found any image to evaluate
    if len(images) == 0:
        print("WARNING: No images found in the test set.")
        return []

    print(f"Starting experiment: method = {config.get('method', 'tv')} | total images: {len(images)}")

    save_only_first = config.get("save_only_first_image", False)
    verbose = config.get("verbose", False)
    image_saved = False

    # Loop through all sinogram test definitions
    for test in config["tests"]:
        sinogram_path = test["sinogram_path"]
        angles = np.linspace(*test["angle_range"], test["num_angles"])
        print(f"  - Processing sinogram: {os.path.basename(sinogram_path)}")

        # Loop through all images and all lambda values
        for gt_path in images:
            for lam in config["lambda_values"]:
                # Generate filename for saving
                out_name = f'{os.path.splitext(os.path.basename(sinogram_path))[0]}_' + \
                           f'{os.path.splitext(os.path.basename(gt_path))[0]}_lambda{str(lam).replace(".", "")}.png'

                # Save only the first image if flag is active
                if config.get("save_images", True) and (not save_only_first or not image_saved):
                    save_path = os.path.join(config["reconstruction_folder"], out_name)
                    image_saved = True
                else:
                    save_path = None

                # Run evaluation and store results
                res = evaluate_single_reconstruction(
                    gt_path, sinogram_path, angles,
                    method=config.get("method", "tv"),
                    lambda_tv=lam,
                    iterations=config.get("iterations", 300),
                    save_path=save_path,
                    verbose=verbose
                )
                results.append(res)

    print(f"Experiment completed: total reconstructions = {len(results)}")
    return results

def save_results_to_csv(results, path):
    """
    Saves all experiment results to a CSV file.

    Parameters:
    - results (list of dict): The list of results to write.
    - path (str): The output file path.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Results saved to: {path}")
