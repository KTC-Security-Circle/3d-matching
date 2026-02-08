"""PLYファイルの読み込みと前処理を担当するモジュール。

PLY形式の3D点群ファイルを読み込み、レジストレーションに必要な前処理を行う:
    - ボクセルダウンサンプリング（点群の間引き）
    - 法線推定（ICPのPoint-to-Planeに必要）
    - FPFH特徴量計算（RANSACの特徴量マッチングに必要）
    - ガウシアンノイズの付加（ロバスト性テスト用）
"""

from pathlib import Path

import numpy as np
import open3d as o3d

from utils.setup_logging import setup_logging

logger = setup_logging(__name__)


class Ply:
    """PLYファイルを読み込み、レジストレーション用に前処理するラッパークラス。

    Attributes:
        path: PLYファイルのパス
        pcd: フル解像度の点群（ICPで使用）
        pcd_down: ダウンサンプル済み点群（RANSACで使用）
        pcd_fpfh: FPFH特徴量（Fast Point Feature Histogram）。
                  特徴量ベースのレジストレーションで対応点を見つけるために使用
    """

    def __init__(self, path: Path, voxel_size: float = 0.3) -> None:
        """PLYファイルを読み込み、前処理を実行する。

        Args:
            path: PLYファイルのパス
            voxel_size: ボクセルダウンサンプリングのサイズ（デフォルト: 0.3）。
                       値が大きいほど点群が粗くなるが処理が高速化する。

        Raises:
            FileNotFoundError: 指定パスにファイルが存在しない場合
            TypeError: ファイル拡張子が .ply でない場合
        """
        self.path = path
        if not self.path.exists():
            msg = f"Ply file not found: {self.path}"
            raise FileNotFoundError(msg)
        if self.path.suffix.lower() != ".ply":
            msg = f"File is not a ply file: {self.path}"
            raise TypeError(msg)

        # フル解像度の点群を読み込み
        self.pcd = self._load(self.path)

        # ダウンサンプル + FPFH特徴量の計算
        self.pcd_down, self.pcd_fpfh = self._preprocess(self.pcd, voxel_size)

        # ダウンサンプル済み点群にガウシアンノイズを付加（標準偏差 0.05）
        # ロバスト性テスト: ノイズがある状況でもレジストレーションが機能するか検証するため
        noise = 0.05 * np.random.randn(*np.asarray(self.pcd_down.points).shape)
        self.pcd_down.points = o3d.utility.Vector3dVector(np.asarray(self.pcd_down.points) + noise)

        # フル解像度の点群にも法線を推定（ICPのPoint-to-Planeに必要）
        self._add_normals(self.pcd, voxel_size)
        logger.info("Successfully loaded and preprocessed ply file: %s", self.path)

    def _load(self, path: Path) -> o3d.geometry.PointCloud:
        """PLYファイルからOpen3Dの点群オブジェクトを読み込む。

        Args:
            path: PLYファイルのパス

        Returns:
            読み込まれた点群オブジェクト

        Raises:
            ValueError: 点群が空（点数0）の場合
        """
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
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """点群のダウンサンプリングとFPFH特徴量の計算を行う。

        処理手順:
            1. ボクセルダウンサンプリング: 指定サイズのボクセルで点群を間引く
            2. 法線推定: KDTree近傍探索（半径 = voxel_size * 2, 最大30近傍点）
            3. FPFH特徴量計算: KDTree近傍探索（半径 = voxel_size * 5, 最大100近傍点）

        Args:
            pcd: 元の点群
            voxel_size: ボクセルサイズ

        Returns:
            tuple: (ダウンサンプル済み点群, FPFH特徴量)
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)
        print(np.asarray(pcd_down.points).shape[0])

        # 法線推定: FPFH特徴量の計算に法線が必要
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
        )

        # FPFH (Fast Point Feature Histogram) 特徴量の計算
        # 各点の局所的な幾何学的特徴を33次元のヒストグラムとして表現
        # レジストレーション時の対応点探索に利用される
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        return pcd_down, pcd_fpfh

    def _add_normals(self, pcd: o3d.geometry.PointCloud, voxel_size: float) -> None:
        """点群に法線を推定・付与する。

        ICPのPoint-to-Plane距離メトリックで法線情報が必要なため、
        フル解像度の点群にも法線を推定する。

        Args:
            pcd: 法線を推定する点群
            voxel_size: 法線推定の探索半径の基準サイズ
        """
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
        )


if __name__ == "__main__":
    from pathlib import Path

    # voxel_size = 0.01
    src_path = Path.cwd() / "3d_data" / "sample.ply"
    tgt_path = Path.cwd() / "3d_data" / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    logger.info("Source PLY: %s", src_ply.path)
    logger.info("Target PLY: %s", tgt_ply.path)
