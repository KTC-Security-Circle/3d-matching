from pathlib import Path

import open3d as o3d

from utils.setup_logging import setup_logging

logger = setup_logging(__name__)


class Ply:
    def __init__(self, path: Path, voxel_size: float) -> None:
        self.path = path
        if not self.path.exists():
            msg = f"Ply file not found: {self.path}"
            raise FileNotFoundError(msg)
        if self.path.suffix.lower() != ".ply":
            msg = f"File is not a ply file: {self.path}"
            raise TypeError(msg)

        self.pcd = self._load(self.path)
        self.pcd_down, self.pcd_fpfh = self._preprocess(self.pcd, voxel_size)
        self._add_normals(self.pcd, voxel_size)
        logger.info("Successfully loaded and preprocessed ply file: %s", self.path)

    def _load(self, path: Path) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_points():
            msg = f"Point cloud is empty: {path}"
            logger.error(msg)
            raise ValueError(msg)
        return pcd

    def _preprocess(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
        )
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        return pcd_down, pcd_fpfh

    def _add_normals(self, pcd: o3d.geometry.PointCloud, voxel_size: float) -> None:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
        )


if __name__ == "__main__":
    from pathlib import Path

    voxel_size = 0.01
    src_path = Path.cwd() / "3d_data" / "sample.ply"
    tgt_path = Path.cwd() / "3d_data" / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    logger.info("Source PLY: %s", src_ply.path)
    logger.info("Target PLY: %s", tgt_ply.path)
