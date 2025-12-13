from numpy import ndarray
from open3d import pipelines

from ply import Ply


def refine_registration(
    src: Ply,
    tgt: Ply,
    init_trans: ndarray,
    voxel_size: float,
) -> pipelines.registration.RegistrationResult:
    dist_thresh = voxel_size * 0.4
    return pipelines.registration.registration_icp(
        src.pcd,
        tgt.pcd,
        dist_thresh,
        init_trans,
        pipelines.registration.TransformationEstimationPointToPlane(),
    )
