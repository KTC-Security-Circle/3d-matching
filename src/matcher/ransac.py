from open3d import pipelines

from ply import Ply


def global_registration(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 1,
) -> pipelines.registration.RegistrationResult:
    dist_thresh = voxel_size * 1.5
    return pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        False,  # noqa: FBT003,実際の比較ではTrue
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),  # noqa: FBT003
        3,
        [
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
    )
