# Tomographic Reconstruction with Total Variation Regularization

**DISCLAIMER**: This project currently implements **only reconstruction with Total Variation (TV) regularization**. The extension with **Plug-and-Play** methods will be integrated in a later phase.

## Project Description

This project focuses on image reconstruction in computed tomography (CT) starting from simulated sinograms. The goal is to analyze the effectiveness of Total Variation (TV) regularization in the context of limited-angle geometries. The dataset used is based on real images (e.g., Mayo Clinic Dataset), but the entire sinogram acquisition process is simulated.

---

## Requirements

The entire project relies on **ASTRA Toolbox**, which is compatible **only with Conda environments**. To avoid installation issues, it is highly recommended to use a dedicated Conda environment.

### Recommended Setup

```bash
conda create -n tv_ct python=3.10
conda activate tv_ct
pip install -r requirements.txt
```

---

## Execution

To run an experiment:

1. Place `.png` or `.jpg` images in `data/raw/test/mini_batch/`.
2. Run the `main.py` script:
  ```bash
  python main.py
  ```

---

## Execution Flow

### `main.py`
- Loads the configuration from `config_test2.json`.
- Calls the `evaluate_batch` function from the `evaluation` module.
- Saves metrics in `results/tv_metrics_summary.csv`.

### `evaluation.py`
- `evaluate_batch(config)`: runs multiple experiments on combinations of images, sinograms, and lambda values.
- `evaluate_single_reconstruction(...)`: reconstructs a single image with TV and computes RE, PSNR, and SSIM.
- `save_results_to_csv(...)`: saves results to a `.csv` file.

### `sinogram_simulation.py`
- `simulate_sinogram(...)`: generates sinograms from 2D images using ASTRA.
- `generate_and_save_sinograms(...)`: produces clean and noisy variants of sinograms (limited and full-angle).

### `total_variation.py`
- `chambolle_pock_tv_reconstruction(...)`: implements the Chambolle-Pock algorithm for TV, using projection and backprojection defined with ASTRA.

### `utils.py`
- `compute_metrics(...)`: computes RE, PSNR, SSIM.
- `normalize_image(...)`: normalizes images to [0,1].
- `add_noise(...)`: adds Gaussian noise to sinograms.

---

## Experiment Configuration

Two JSON files control the experiment:

- `config_test2.json` and `experiment_config.json`: specify images, sinograms, angles, lambda, iterations, and output directories.
- Four configurations are defined for each image:
  - `limited_clean`: 30 angles from -30° to +30°, no noise
  - `limited_noisy`: 30 angles, with noise (σ = 0.01)
  - `full_clean`: 180 angles from -90° to +90°, no noise
  - `full_noisy`: 180 angles, with noise

---

## Output

- Sinograms are saved in `data/sinograms/`.
- Reconstructions are saved in `data/reconstructions/`.
- Metrics are saved in `results/tv_metrics_summary.csv`.

---

## Future Extensions

The project will be extended to include:
- Plug-and-Play reconstruction with Residual U-Net.
- Comparison between TV and PnP methods on full images and regions of interest.

