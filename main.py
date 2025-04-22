import numpy as np
import os
from skimage.io import imread
from src.sinogram_simulation import generate_and_save_sinograms
from src.total_variation import fbp_reconstruction
from src.utils import compute_metrics, normalize_image
import matplotlib.pyplot as plt

def main():
    # Path immagine di test (puoi modificarlo con una tua immagine)
    image_path = "data/raw/test/C081/1.png"  # ad esempio un'immagine 512x512 del Mayo dataset
    sinogram_folder = "data/sinograms/"
    recon_folder = "data/reconstructions/"
    os.makedirs(recon_folder, exist_ok=True)

    # 1. Simula i sinogrammi
    generate_and_save_sinograms(image_path, sinogram_folder)

    # 2. Carica sinogramma limitato senza rumore
    angles = np.linspace(-30, 30, 30)
    sinogram = np.load(os.path.join(sinogram_folder, "limited_clean_sinogram.npy"))

    # 3. Ricostruzione FBP
    reco = fbp_reconstruction(sinogram, angles)

    # 4. Carica ground truth e normalizza
    gt = imread(image_path, as_gray=True)
    gt = normalize_image(gt)
    gt = np.resize(gt, reco.shape)

    # 5. Calcola metriche
    re, p, s = compute_metrics(gt, reco)
    print(f"RE: {re:.4f}, PSNR: {p:.2f}, SSIM: {s:.3f}")

    # 6. Visualizza risultati
    plt.subplot(1, 2, 1)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reco, cmap="gray")
    plt.title("FBP Reconstruction")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
