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
from matcher.ransac import (
    compute_feature_correspondences,
    compute_step_transformation,  # ★変更
    evaluate_inlier_ratio,
)
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

        self.info_label = o3dv_gui.Label("")
        gui_layout.add_child(self.info_label)

        self.random_transform_button = o3dv_gui.Button("Random Transform")
        gui_layout.add_child(self.random_transform_button)

        self.ransac_button = o3dv_gui.Button("Run RANSAC (Manual Step)")
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
    noise_ratio: float = 2.0


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

        self.source.pcd.transform(transformation)
        self.source.pcd_down.transform(transformation)

        self.source_pcd_orig = copy.deepcopy(self.source.pcd)
        self.source_down_orig = copy.deepcopy(self.source.pcd_down)

        self._update_scene(self.source.pcd)
        self.view_manager.label.text = "Random Transformed"
        self.view_manager.window.post_redraw()

    def _run_ransac_worker(self) -> None:
        if self.settings is None:
            return

        corres = compute_feature_correspondences(self.source, self.target, noise_ratio=self.settings.noise_ratio)

        iter_num = 0
        max_iter = self.settings.ransac_iteration

        best_result = None
        best_fitness = -1.0

        logger.info(f"Start RANSAC: {len(corres)} correspondences (Noise Ratio: {self.settings.noise_ratio})")

        while iter_num < max_iter:
            iter_num += 1

            # ★ 手動計算関数を呼び出す
            # これで必ず何らかの変換行列が返ってくる
            result = compute_step_transformation(self.source, self.target, corres)

            # 評価（Fitness/Inlier Ratio）を計算
            w_current = evaluate_inlier_ratio(
                self.source, self.target, corres, result.transformation, self.settings.voxel_size
            )
            result.fitness = w_current  # 便宜上fitnessに格納

            # ベスト更新
            if best_result is None or w_current > best_fitness:
                best_result = result
                best_fitness = w_current

            # 画面更新
            self.app.post_to_main_thread(
                self.view_manager.window,
                lambda res=result, it=iter_num, val=w_current, b_fit=best_fitness: self._update_viz(
                    res, it, val, b_fit
                ),
            )

            # 速度調整（0.01〜0.05推奨）
            time.sleep(0.03)

        self.last_ransac_result = best_result
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._finalize_ransac(best_result))

    def _update_viz(self, result, iter_num, w, best_fit):
        # 現在の試行結果で動かす
        temp = copy.deepcopy(self.source_pcd_orig)
        temp.transform(result.transformation)
        self._update_scene(temp)

        self.view_manager.label.text = f"Iter: {iter_num}"
        self.view_manager.info_label.text = f"CurFit: {w:.4f} | BestFit: {best_fit:.4f}"
        self.view_manager.window.post_redraw()

    def _finalize_ransac(self, result):
        if result:
            self.source.pcd.transform(result.transformation)
            self.source.pcd_down.transform(result.transformation)
            self.source_pcd_orig = copy.deepcopy(self.source.pcd)
            self.source_down_orig = copy.deepcopy(self.source.pcd_down)
            self._update_scene(self.source.pcd)
            self.view_manager.label.text = f"Done. Final Best: {result.fitness:.4f}"
        else:
            self.view_manager.label.text = "Failed."
        self.view_manager.window.post_redraw()

    def _run_icp_worker(self) -> None:
        if self.settings is None or self.last_ransac_result is None:
            return
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

    visualizer.invoke(
        MatcherSettings(
            voxel_size=voxel_size,
            ransac_iteration=10000,
            noise_ratio=2.0,
        ),
        is_logging=True,
    )
