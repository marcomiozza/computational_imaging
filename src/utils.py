import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_metrics(gt, reco):
    re = np.linalg.norm(reco - gt) / np.linalg.norm(gt)
    psnr_val = psnr(gt, reco, data_range=1.0)
    ssim_val = ssim(gt, reco, data_range=1.0)  
    return re, psnr_val, ssim_val

def normalize_image(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def add_noise(sinogram, sigma=0.01):
    noise = np.random.normal(0, sigma, sinogram.shape)
    return sinogram + noise

