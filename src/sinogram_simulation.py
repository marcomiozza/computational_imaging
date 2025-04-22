import numpy as np
import astra
import os
from skimage.io import imread
from skimage.transform import resize
from src.utils import add_noise, normalize_image

def simulate_sinogram(image, angles_deg, add_noise_flag=False, noise_sigma=0.01):
    image = normalize_image(image)
    image = resize(image, (512, 512), anti_aliasing=True)
    
    # Geometria
    angles_rad = np.deg2rad(angles_deg)
    vol_geom = astra.create_vol_geom(image.shape[0], image.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[0], angles_rad)

    # Crea id per i dati
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    sinogram_id = astra.data2d.create('-sino', proj_geom)
    image_id = astra.data2d.create('-vol', vol_geom, data=image)

    cfg = astra.astra_dict('FP')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = image_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sinogram = astra.data2d.get(sinogram_id)

    # Aggiungi rumore se richiesto
    if add_noise_flag:
        sinogram += add_noise(sinogram, sigma=noise_sigma)

    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(image_id)
    astra.projector.delete(proj_id)

    return sinogram, proj_geom, vol_geom


def generate_and_save_sinograms(image_path, output_folder, noise_sigma=0.01):
    import os
    from skimage.io import imread

    os.makedirs(output_folder, exist_ok=True)
    image = imread(image_path, as_gray=True)

    configs = [
        ("limited_clean", np.linspace(-30, 30, 30), False),
        ("limited_noisy", np.linspace(-30, 30, 30), True),
        ("full_clean", np.linspace(-90, 90, 180), False),
        ("full_noisy", np.linspace(-90, 90, 180), True),
    ]

    for name, angles, noise in configs:
        sinogram, _, _ = simulate_sinogram(image, angles, add_noise_flag=noise, noise_sigma=noise_sigma)
        np.save(os.path.join(output_folder, f"{name}_sinogram.npy"), sinogram)
