import os
import sys
import numpy as np
import torch
import csv

# Setup path IPPy
sys.path.append("/content/COMPUTATIONAL_IMAGING")
from IPPy import operators, solvers, utilities
from IPPy.utilities import load_image, save_image, create_path_if_not_exists, metrics

# Percorsi
image_path = "/content/COMPUTATIONAL_IMAGING/data/test/0.png"
output_dir = "/content/output"
create_path_if_not_exists(output_dir)

# Parametri
image_size = 256
detector_size = 512
geometry = "parallel"
noise_level = 0.01  # Rumore gaussiano (1%)
lmbda = 0.001
maxiter = 300
p = 1
device = torch.device(utilities.get_device())

# Imposta intervallo di angoli in gradi
start_angle_deg = 0    # es: -30, 0, -90, ...
end_angle_deg = 360       # es: 30, 180, 90, 360, ...
n_angles = 360   



# Carica immagine e ridimensiona
x_true = load_image(image_path).to(device)
x_true = torch.nn.functional.interpolate(x_true, size=(image_size, image_size), mode="bilinear")

# Salva immagine originale
save_image(x_true.detach().cpu(), os.path.join(output_dir, "original.png"))




# Conversione in radianti
angles = np.linspace(
    np.deg2rad(start_angle_deg),
    np.deg2rad(end_angle_deg),
    n_angles,
    endpoint=False
)

# Creazione dell’operatore CT con angoli personalizzati
K = operators.CTProjector(
    img_shape=(image_size, image_size),
    angles=angles,
    det_size=detector_size,
    geometry=geometry,
)

# Calcola sinogramma pulito
y_clean = K(x_true).detach()

# Aggiungi rumore gaussiano
noise = utilities.gaussian_noise(y_clean, noise_level=noise_level)
y_noisy = y_clean + noise

# Salva sinogrammi
save_image(y_clean.detach().cpu(), os.path.join(output_dir, "sinogram_clean.png"))
save_image(y_noisy.detach().cpu(), os.path.join(output_dir, "sinogram_noisy.png"))

# Ricostruzione da sinogramma rumoroso
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

# Salvataggio metriche in CSV
csv_path = os.path.join(output_dir, "metrics.csv")
header = ["filename", "RE", "PSNR", "SSIM"]
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([os.path.basename(image_path), RE, PSNR, SSIM])

# Output
print(f"Terminato: {os.path.basename(image_path)}")
print(f"  RE   = {RE:.4f}")
print(f"  PSNR = {PSNR:.2f}")
print(f"  SSIM = {SSIM:.4f}")
print("-" * 40)
