import astra
import numpy as np

def fbp_reconstruction(sinogram, angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    proj_geom = astra.create_proj_geom('parallel', 1.0, sinogram.shape[1], angles_rad)
    vol_geom = astra.create_vol_geom(sinogram.shape[1], sinogram.shape[1])

    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

   
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)

    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = projector_id  

    cfg['option'] = {'FilterType': 'Ram-Lak'}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(rec_id)

    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(rec_id)
    astra.projector.delete(projector_id)

    return reconstruction
