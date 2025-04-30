import os
import sys
import numpy as np
import torch

# Setup path IPPy (modifica se il path cambia)
sys.path.append("/content/COMPUTATIONAL_IMAGING")
from IPPy import operators, solvers, utilities
from IPPy.utilities import load_image, save_image, show, create_path_if_not_exists

# Percorsi (modifica se usi Drive)
image_path = "/content/COMPUTATIONAL_IMAGING/data/test/0.png"
output_dir = "/content/output"
create_path_if_not_exists(output_dir)

# Parametri
image_size = 256
detector_size = 512
lmbda = 0.001
maxiter = 300
p = 1
geometry = "parallel"
device = torch.device(utilities.get_device())

# Carica immagine e ridimensiona
x_true = load_image(image_path).to(device)
x_true = torch.nn.functional.interpolate(x_true, size=(image_size, image_size), mode="bilinear")

# Crea operatore CT da -30° a 30°
angles = np.linspace(-30 * np.pi / 180, 30 * np.pi / 180, 180, endpoint=False)
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
    verbose=True,
)

# Salvataggio risultati
save_image(x_true.detach().cpu(), os.path.join(output_dir, "original.png"))
save_image(y.detach().cpu(), os.path.join(output_dir, "sinogram.png"))
save_image(x_rec.detach().cpu(), os.path.join(output_dir, "reconstruction.png"))

# Visualizza i risultati
show(
    [x_true.detach().cpu(), y.detach().cpu(), x_rec.detach().cpu()],
    title=["Original", "Sinogram", "Reconstruction"],
    save_path=output_dir,
)
