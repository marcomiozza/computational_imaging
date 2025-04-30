import os
import numpy as np
import torch
from IPPy.operators import CTProjector
from IPPy.utilities import load_image, save_image, create_path_if_not_exists

# Config
input_dir = "data/raw/test/mini_batch"
output_dir = "data/sinograms"
create_path_if_not_exists(output_dir)

# Angoli
angles = np.linspace(0, 2*np.pi,180, endpoint=False)

# Lista immagini
image_files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith(".png")
])

# Loop sulle immagini
for img_path in image_files:
    x = load_image(img_path)
    _, _, nx, ny = x.shape

    # Costruisci il proiettore CORRETTO
    P = CTProjector(
        img_shape=(nx, ny),
        angles=angles,
        det_size=2*max(nx, ny),   # 512 se 256x256
        geometry="parallel"       # << OBBLIGATORIO mettere "parallel"
    )

    # Genera il sinogramma
    sino = P(x)

    # Salva il sinogramma
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    torch.save(sino, os.path.join(output_dir, f"{base_name}_sino.pt"))

    print(f"Salvato sinogramma per {base_name}")

print("Tutto fatto!")
