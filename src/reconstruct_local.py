import os
import sys
import numpy as np
import torch

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from IPPy import operators, solvers, utilities
from IPPy.utilities import load_image, save_image, show, create_path_if_not_exists

# Parametri
image_path = "data/raw/test/mini_batch/0.png"  
image_size = 256
detector_size = 512
lmbda = 0.001
maxiter = 300
p = 1
geometry = "parallel"
device = utilities.get_device()

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
y = K(x_true)
y = y.detach()  
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

# Crea cartella output
create_path_if_not_exists("output")

# Salva immagini
save_image(x_true.detach().cpu(), "output/original.png")
save_image(y.detach().cpu(), "output/sinogram.png")
save_image(x_rec.detach().cpu(), "output/reconstruction.png")

# Mostra risultati
show([x_true.detach().cpu(), y.detach().cpu(), x_rec.detach().cpu()],
     title=["Original", "Sinogram", "Reconstruction"])
