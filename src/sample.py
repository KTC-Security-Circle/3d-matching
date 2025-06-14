import copy

import numpy as np
import open3d as o3d


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    print(f"Loaded '{path}', #points:", np.asarray(pcd.points).shape[0])
    return pcd


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        dist_thresh,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )
    print("Global RANSAC result:", result)
    return result.transformation


def refine_registration(src, tgt, init_trans, voxel_size):
    dist_thresh = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, dist_thresh, init_trans, o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    print("ICP refinement:", result)
    return result.transformation


if __name__ == "__main__":
    import os

    voxel_size = 0.01
    src = load_pcd(os.getcwd() + "/3d_data/sample.ply")
    tgt = load_pcd(os.getcwd() + "/3d_data/target.ply")
    src_down, src_fpfh = preprocess(src, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt, voxel_size)

    # --- 法線推定を追加 ---
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # ----------------------

    init_trans = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)  # RANSAC
    final_trans = refine_registration(src, tgt, init_trans, voxel_size)  # ICP
