from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import open3d as o3d
import open3d.visualization.gui as o3dv_gui
import open3d.visualization.rendering as o3dv_rendering

from matcher.icp import refine_registration

# ★変更: ransac.py の新しい関数をインポート
from matcher.ransac import global_registration, compute_feature_correspondences, run_ransac_step, evaluate_inlier_ratio
from utils.setup_logging import setup_logging

if TYPE_CHECKING:
    from ply import Ply

logger = setup_logging(__name__)

SOURCE_NAME = "source"
TARGET_NAME = "target"


@dataclass
class MatcherGeometyData:
    source: Ply
    target: Ply


class ViewManager:
    def __init__(self, app: o3dv_gui.Application, window_name: str, init_data: MatcherGeometyData) -> None:
        self.window = app.create_window(window_name, 1024, 768)

        self.scene = o3dv_rendering.Open3DScene(self.window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.material.point_size = 5.0
        self.scene.add_geometry(SOURCE_NAME, init_data.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, init_data.target.pcd, self.material)

        em = self.window.theme.font_size
        gui_layout = o3dv_gui.Vert(0, o3dv_gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        gui_layout.frame = o3dv_gui.Rect(
            self.window.content_rect.x,
            self.window.content_rect.y,
            250,
            self.window.content_rect.height,
        )

        self.label = o3dv_gui.Label("Ready")
        gui_layout.add_child(self.label)

        # 情報表示用ラベルを追加
        self.info_label = o3dv_gui.Label("")
        gui_layout.add_child(self.info_label)

        self.random_transform_button = o3dv_gui.Button("Random Transform")
        gui_layout.add_child(self.random_transform_button)

        self.ransac_button = o3dv_gui.Button("Run RANSAC Step-by-Step")
        gui_layout.add_child(self.ransac_button)

        self.icp_button = o3dv_gui.Button("Run ICP")
        gui_layout.add_child(self.icp_button)

        scene_widget = o3dv_gui.SceneWidget()
        scene_widget.scene = self.scene
        scene_widget.setup_camera(60.0, self.scene.bounding_box, self.scene.bounding_box.get_center())
        scene_widget.frame = o3dv_gui.Rect(
            gui_layout.frame.get_right(),
            self.window.content_rect.y,
            self.window.content_rect.width - gui_layout.frame.width,
            self.window.content_rect.height,
        )

        self.window.add_child(gui_layout)
        self.window.add_child(scene_widget)


@dataclass
class MatcherSettings:
    voxel_size: float
    ransac_iteration: int
    noise_ratio: float = 20.0  # ★正解の20倍のゴミを混ぜる（これなら絶対一発では通りません）


class VisualizeMatcher:
    RANDOM_ROTATION_RANGE_RAD = (-np.pi / 6, np.pi / 6)
    RANDOM_TRANSLATION_RANGE = (-0.1, 0.1)

    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        self.source = source
        self.target = target
        self.window_name = window_name
        self.settings: MatcherSettings | None = None
        self.is_logging = False
        self.last_ransac_result = None
        self.source_base_center = np.asarray(self.source.pcd.get_center())
        self.rng = np.random.default_rng()

        # ★初期状態を保存（重要）
        self.source_pcd_orig = copy.deepcopy(self.source.pcd)
        self.source_down_orig = copy.deepcopy(self.source.pcd_down)

        self.app = o3dv_gui.Application.instance
        self.app.initialize()
        self.view_manager = ViewManager(
            self.app,
            self.window_name,
            MatcherGeometyData(source=self.source, target=self.target),
        )

        self.view_manager.random_transform_button.set_on_clicked(self._on_random_transform)
        self.view_manager.ransac_button.set_on_clicked(self._on_run_ransac)
        self.view_manager.icp_button.set_on_clicked(self._on_run_icp)

    def invoke(self, settings: MatcherSettings, *, is_logging: bool) -> None:
        self.settings = settings
        self.is_logging = is_logging
        self.view_manager.window.post_redraw()
        self.app.run()

    def _on_run_ransac(self) -> None:
        if self.settings is None:
            return
        self.view_manager.label.text = "Initializing..."
        self.view_manager.window.post_redraw()
        self.app.run_in_thread(self._run_ransac_worker)

    def _on_run_icp(self) -> None:
        if self.settings is None or self.last_ransac_result is None:
            self.view_manager.label.text = "Run RANSAC first!"
            return
        self.view_manager.label.text = "Running ICP..."
        self.view_manager.window.post_redraw()
        self.app.run_in_thread(self._run_icp_worker)

    def _on_random_transform(self) -> None:
        angles = self.rng.uniform(*self.RANDOM_ROTATION_RANGE_RAD, 3)
        # (回転行列生成省略: 既存コードと同じ)
        rx = np.array(
            [[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]]
        )
        ry = np.array(
            [[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]]
        )
        rz = np.array(
            [[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]]
        )
        rotation = rz @ ry @ rx
        translation = self.rng.uniform(*self.RANDOM_TRANSLATION_RANGE, 3)

        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        offset = -rotation @ self.source_base_center + self.source_base_center + translation
        transformation[:3, 3] = offset

        # 本体を変換
        self.source.pcd.transform(transformation)
        self.source.pcd_down.transform(transformation)

        # ★ここが重要：Origも今の位置でリセットする（ここからRANSACを開始するため）
        self.source_pcd_orig = copy.deepcopy(self.source.pcd)
        self.source_down_orig = copy.deepcopy(self.source.pcd_down)

        self._update_scene(self.source.pcd)
        self.view_manager.label.text = "Random Transformed"
        self.view_manager.window.post_redraw()

    def _run_ransac_worker(self) -> None:
        """ステップ実行用のRANSACワーカー"""
        if self.settings is None:
            return

        # 1. 対応点リスト作成（ここで大量のノイズを混ぜる！）
        corres = compute_feature_correspondences(self.source, self.target, noise_ratio=self.settings.noise_ratio)

        iter_num = 0
        max_iter = self.settings.ransac_iteration
        best_result = None

        # ログ
        logger.info(f"Start RANSAC: {len(corres)} correspondences (Noise Ratio: {self.settings.noise_ratio})")

        while iter_num < max_iter:
            iter_num += 1

            # 2. ★超重要★ 1回だけ試行する (max_iteration=1)
            # そして、self.sourceではなく、動かしていない self.source...orig を使う概念だが、
            # registration_ransac_based_on_correspondence は座標値そのものを使うため、
            # ソース点群を物理的に動かしてはいけない。
            result = run_ransac_step(
                self.source,  # 座標はずっと初期位置のまま
                self.target,
                corres,
                self.settings.voxel_size,
                max_iteration=1,  # ←これが1000だと一発で終わります。必ず1にする。
            )

            # 3. ベスト更新チェック
            is_better = False
            if best_result is None:
                is_better = True
            elif result.fitness > best_result.fitness:
                is_better = True

            # ベスト更新時のみ描画
            if is_better:
                best_result = result

                # インライア率wの計算（早期終了判定用）
                w = evaluate_inlier_ratio(
                    self.source, self.target, corres, result.transformation, self.settings.voxel_size
                )

                # GUI更新
                self.app.post_to_main_thread(
                    self.view_manager.window, lambda res=best_result, it=iter_num, val=w: self._update_viz(res, it, val)
                )

                # 少し待つ（可視化演出）
                time.sleep(0.01)

        # 最後に確定
        self.last_ransac_result = best_result
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._finalize_ransac(best_result))

    def _update_viz(self, result, iter_num, w):
        # 元の点群（orig）のコピーを変換して表示
        # ソース自体(self.source)はいじらない
        temp = copy.deepcopy(self.source_pcd_orig)
        temp.transform(result.transformation)

        self._update_scene(temp)
        self.view_manager.label.text = f"Iter: {iter_num}"
        self.view_manager.info_label.text = f"Fit: {result.fitness:.4f} | w: {w:.3f}"
        self.view_manager.window.post_redraw()

    def _finalize_ransac(self, result):
        if result:
            # ここで初めてソース本体を動かす
            self.source.pcd.transform(result.transformation)
            self.source.pcd_down.transform(result.transformation)

            # Origも更新（ここがICPのスタート地点になる）
            self.source_pcd_orig = copy.deepcopy(self.source.pcd)
            self.source_down_orig = copy.deepcopy(self.source.pcd_down)

            self._update_scene(self.source.pcd)
            self.view_manager.label.text = f"Done. Fitness: {result.fitness:.4f}"
        else:
            self.view_manager.label.text = "Failed."
        self.view_manager.window.post_redraw()

    def _run_icp_worker(self) -> None:
        if self.settings is None or self.last_ransac_result is None:
            return
        # ICPは既存のまま
        res = refine_registration(self.source, self.target, np.eye(4), self.settings.voxel_size)
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._finalize_icp(res))

    def _finalize_icp(self, result):
        self.source.pcd.transform(result.transformation)
        self.source.pcd_down.transform(result.transformation)
        self._update_scene(self.source.pcd)
        self.view_manager.label.text = f"ICP Done. Fit: {result.fitness:.4f}"
        self.view_manager.window.post_redraw()

    def _update_scene(self, pcd):
        if self.view_manager.scene.has_geometry(SOURCE_NAME):
            self.view_manager.scene.remove_geometry(SOURCE_NAME)
        self.view_manager.scene.add_geometry(SOURCE_NAME, pcd, self.view_manager.material)


if __name__ == "__main__":
    from pathlib import Path
    from ply import Ply

    voxel_size = 0.3
    base_path = Path(__file__).parent.parent.parent / "3d_data"
    src_path = base_path / "sample.ply"
    tgt_path = base_path / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    visualizer = VisualizeMatcher(src_ply, tgt_ply)

    # ★ noise_ratio=20.0 (20倍のゴミ) を設定
    # ★ ransac_iteration は試行回数の上限（例: 50000回）
    visualizer.invoke(
        MatcherSettings(voxel_size=voxel_size, ransac_iteration=50000, noise_ratio=20.0),
        is_logging=True,
    )
