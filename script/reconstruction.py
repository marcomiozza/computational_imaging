import os
import sys
import numpy as np
import torch
import csv

# Setup path IPPy (modifica se il path cambia)
sys.path.append("/content/COMPUTATIONAL_IMAGING")
from IPPy import operators, solvers, utilities
from IPPy.utilities import load_image, save_image, create_path_if_not_exists, metrics

# Percorsi (modifica se usi Drive)
image_path = "/content/COMPUTATIONAL_IMAGING/data/test/0.png"
output_dir = "/content/output"
create_path_if_not_exists(output_dir)

# Parametri
image_size = 256
detector_size = 512
lmbda = 0.01
maxiter = 300
p = 1
geometry = "parallel"
device = torch.device(utilities.get_device())

# Carica immagine e ridimensiona
x_true = load_image(image_path).to(device)
print(f"Elaborazione: {os.path.basename(image_path)}")

x_true = torch.nn.functional.interpolate(x_true, size=(image_size, image_size), mode="bilinear")

# Crea operatore CT da 0° a 360°
angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
K = operators.CTProjector(
    img_shape=(image_size, image_size),
    angles=angles,
    det_size=detector_size,
    geometry=geometry,
)

# Calcola sinogramma
y = K(x_true).detach()

# Ricostruzione TV con Chambolle-Pock unconstrained
solver = solvers.ChambollePockTpVUnconstrained(K)
x_rec, info = solver(
    y,
    lmbda=lmbda,
    x_true=x_true,
    starting_point=None,
    maxiter=maxiter,
    p=p,
    verbose=False,
)

# Salvataggio immagini
save_image(x_true.detach().cpu(), os.path.join(output_dir, "original.png"))
save_image(y.detach().cpu(), os.path.join(output_dir, "sinogram.png"))
save_image(x_rec.detach().cpu(), os.path.join(output_dir, "reconstruction.png"))

# Calcolo metriche finali
RE = metrics.RE(x_rec, x_true).item()  
PSNR = metrics.PSNR(x_rec, x_true)    
SSIM = metrics.SSIM(x_rec, x_true)     




# Salvataggio metriche in CSV
csv_path = os.path.join(output_dir, "metrics.csv")
header = ["filename", "RE", "PSNR", "SSIM"]

# Scrivi intestazione se il file non esiste
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([os.path.basename(image_path), RE, PSNR, SSIM])

# Messaggio a schermo
print(f"Terminato: {os.path.basename(image_path)}")
print(f"  RE   = {RE:.4f}")
print(f"  PSNR = {PSNR:.2f}")
print(f"  SSIM = {SSIM:.4f}")
print("-" * 40)
