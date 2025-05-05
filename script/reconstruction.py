import os
import sys
import glob
import numpy as np
import torch
import csv

# Setup IPPy path
sys.path.append("/content/COMPUTATIONAL_IMAGING")
from IPPy import operators, solvers, utilities
from IPPy.utilities import metrics, load_image, create_path_if_not_exists

# Parametri generali
image_dir = "/content/COMPUTATIONAL_IMAGING/data/test"
output_csv = "/content/output3/batch_results.csv"
image_size = 256
detector_size = 512
geometry = "parallel"
lmbda = 0.001
maxiter = 300
p = 1
device = torch.device(utilities.get_device())

# Configurazione fissa
start_deg, end_deg = -90, 90
n_angles = abs(end_deg - start_deg)
noise_levels = [0.0, 0.01]

# Trova immagini PNG nella cartella
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
print(f"Numero immagini trovate: {len(image_paths)}")

# Conversione angoli in radianti
angles = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), n_angles, endpoint=False)

# Operatore CT
K = operators.CTProjector(
    img_shape=(image_size, image_size),
    angles=angles,
    det_size=detector_size,
    geometry=geometry,
)

# Solver
solver = solvers.ChambollePockTpVUnconstrained(K)

# Prepara cartella output
create_path_if_not_exists(os.path.dirname(output_csv))

# Scrivi intestazione CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "noise_level", "lambda", "RE", "PSNR", "SSIM"])

# Loop su tutte le immagini
for idx, image_path in enumerate(image_paths):
    filename = os.path.basename(image_path)
    x_true = load_image(image_path).to(device)
    x_true = torch.nn.functional.interpolate(x_true, size=(image_size, image_size), mode="bilinear")

    y_clean = K(x_true).detach()

    for noise_level in noise_levels:
        noise = utilities.gaussian_noise(y_clean, noise_level=noise_level)
        y_noisy = y_clean + noise

        x_rec, _ = solver(
            y_noisy,
            lmbda=lmbda,
            x_true=x_true,
            starting_point=None,
            maxiter=maxiter,
            p=p,
            verbose=False,
        )

        RE = metrics.RE(x_rec, x_true).item()
        PSNR = metrics.PSNR(x_rec, x_true)
        SSIM = metrics.SSIM(x_rec, x_true)

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, noise_level, lmbda, RE, PSNR, SSIM])

        print(f"[{idx+1}/{len(image_paths)}] {filename} | noise={noise_level} | RE={RE:.4f}, PSNR={PSNR:.2f}, SSIM={SSIM:.4f}")
