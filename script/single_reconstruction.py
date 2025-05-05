import os
import sys
import numpy as np
import torch
import csv

# Setup IPPy path
sys.path.append("/content/COMPUTATIONAL_IMAGING")
from IPPy import operators, solvers, utilities
from IPPy.utilities import load_image, save_image, create_path_if_not_exists, metrics

# Parametri generali
image_path = "/content/COMPUTATIONAL_IMAGING/data/test/0.png"
base_output_dir = "/content/output2"
image_size = 256
detector_size = 512
geometry = "parallel"
lmbda = 0.001
maxiter = 300
p = 1
device = torch.device(utilities.get_device())

# Configurazioni da testare
angle_configs = [(-30, 30), (-90, 90), (0, 360)]
noise_levels = [0.0, 0.01]

# Carica e ridimensiona l'immagine
x_true = load_image(image_path).to(device)
x_true = torch.nn.functional.interpolate(x_true, size=(image_size, image_size), mode="bilinear")

# Loop su tutti i test
for start_deg, end_deg in angle_configs:
    n_angles = abs(end_deg - start_deg)  # 1 angolo per grado

    for noise_level in noise_levels:
        # Define path per salvataggio
        test_name = f"angles_{start_deg}_{end_deg}_noise_{noise_level:.2f}_lambda_{lmbda:.3f}"
        output_dir = os.path.join(base_output_dir, test_name)
        create_path_if_not_exists(output_dir)

        print(f"\nTest: {test_name}")
        print(f"   -> Angoli: {n_angles} tra {start_deg}° e {end_deg}°")
        print(f"   -> Noise: {noise_level}")

        # Salva immagine originale
        save_image(x_true.detach().cpu(), os.path.join(output_dir, "original.png"))

        # Conversione angoli in radianti
        angles = np.linspace(
            np.deg2rad(start_deg),
            np.deg2rad(end_deg),
            n_angles,
            endpoint=False
        )

        # Crea operatore CT
        K = operators.CTProjector(
            img_shape=(image_size, image_size),
            angles=angles,
            det_size=detector_size,
            geometry=geometry,
        )

        # Calcola sinogramma
        y_clean = K(x_true).detach()
        noise = utilities.gaussian_noise(y_clean, noise_level=noise_level)
        y_noisy = y_clean + noise

        # Salva sinogrammi
        save_image(y_clean.detach().cpu(), os.path.join(output_dir, "sinogram_clean.png"))
        save_image(y_noisy.detach().cpu(), os.path.join(output_dir, "sinogram_noisy.png"))

        # Ricostruzione
        solver = solvers.ChambollePockTpVUnconstrained(K)
        x_rec, info = solver(
            y_noisy,
            lmbda=lmbda,
            x_true=x_true,
            starting_point=None,
            maxiter=maxiter,
            p=p,
            verbose=False,
        )

        # Salva ricostruzione
        save_image(x_rec.detach().cpu(), os.path.join(output_dir, "reconstruction.png"))

        # Calcolo metriche
        RE = metrics.RE(x_rec, x_true).item()
        PSNR = metrics.PSNR(x_rec, x_true)
        SSIM = metrics.SSIM(x_rec, x_true)

        # Salva metriche
        csv_path = os.path.join(output_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "RE", "PSNR", "SSIM"])
            writer.writerow([os.path.basename(image_path), RE, PSNR, SSIM])

        # Output su console
        print(f"   -> RE   = {RE:.4f}")
        print(f"   -> PSNR = {PSNR:.2f}")
        print(f"   -> SSIM = {SSIM:.4f}")
        print("-" * 50)
