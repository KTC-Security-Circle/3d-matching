from numpy import ndarray
from open3d import pipelines

from ply import Ply
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)


def refine_registration(
    src: Ply,
    tgt: Ply,
    init_trans: ndarray,
    voxel_size: float,
) -> ndarray:
    dist_thresh = voxel_size * 0.4
    result = pipelines.registration.registration_icp(
        src.pcd,
        tgt.pcd,
        dist_thresh,
        init_trans,
        pipelines.registration.TransformationEstimationPointToPlane(),
    )
    logger.info("ICP refinement: %s", result)
    return result.transformation
