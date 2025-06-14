from numpy import ndarray
from open3d import pipelines

from ply import Ply
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)


def global_registration(src: Ply, tgt: Ply, voxel_size: float) -> ndarray:
    dist_thresh = voxel_size * 1.5
    result = pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        True,  # noqa: FBT003
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),  # noqa: FBT003
        3,
        [
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )
    logger.info("Global RANSAC result: %s", result)
    return result.transformation
