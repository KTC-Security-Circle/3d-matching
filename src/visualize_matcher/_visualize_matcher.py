from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import open3d as o3d
import open3d.visualization.gui as o3dv_gui  # pyright: ignore[reportMissingImports]
import open3d.visualization.rendering as o3dv_rendering  # pyright: ignore[reportMissingImports]

from matcher.icp import refine_registration
from matcher.ransac import global_registration
from utils.setup_logging import setup_logging

if TYPE_CHECKING:
    from ply import Ply

logger = setup_logging(__name__)


class VisualzerProtocol(Protocol):
    def update_geometry(self, geometry: o3d.geometry.Geometry) -> None: ...

    def poll_events(self) -> None: ...

    def update_renderer(self) -> None: ...


class VisualizeInfoProtocol(Protocol):
    fitness: float
    inlier_rmse: float


SOURCE_NAME = "source"
TARGET_NAME = "target"


@dataclass
class MatcherGeometyData:
    source: Ply
    target: Ply


class ViewManager:
    def __init__(self, app: o3dv_gui.Application, window_name: str, init_data: MatcherGeometyData) -> None:
        self.window = app.create_window(window_name, 800, 600)

        # ==== Scene / Material ====
        self.scene = o3dv_rendering.Open3DScene(self.window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.scene.add_geometry(SOURCE_NAME, init_data.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, init_data.target.pcd, self.material)

        # ==== 左側 GUI レイアウト ====
        em = self.window.theme.font_size
        gui_layout = o3dv_gui.Vert(0, o3dv_gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        # 左側の幅を 250px に決め打ち
        gui_layout.frame = o3dv_gui.Rect(
            self.window.content_rect.x,
            self.window.content_rect.y,
            250,
            self.window.content_rect.height,
        )

        self.label = o3dv_gui.Label("RANSAC Fitness: progressing...")
        gui_layout.add_child(self.label)

        # ==== ランダム変換ボタン ====
        self.random_transform_button = o3dv_gui.Button("Random")
        gui_layout.add_child(self.random_transform_button)

        # ==== RANSAC ボタン ====
        self.ransac_button = o3dv_gui.Button("Run RANSAC")
        gui_layout.add_child(self.ransac_button)

        # ==== ICP ボタン ====
        self.icp_button = o3dv_gui.Button("Run ICP")
        gui_layout.add_child(self.icp_button)

        # ==== 右側 SceneWidget ====
        scene_widget = o3dv_gui.SceneWidget()
        scene_widget.scene = self.scene
        scene_widget.setup_camera(
            60.0,
            self.scene.bounding_box,
            self.scene.bounding_box.get_center(),
        )

        scene_widget.frame = o3dv_gui.Rect(
            gui_layout.frame.get_right(),
            self.window.content_rect.y,
            self.window.content_rect.width - gui_layout.frame.width,
            self.window.content_rect.height,
        )

        # ==== Window に直接 add ====
        self.window.add_child(gui_layout)
        self.window.add_child(scene_widget)


@dataclass
class MatcherSettings:
    voxel_size: float
    ransac_iteration: int


class VisualizeMatcher:
    # ランダム変換のパラメーター（後から調整可能）
    RANDOM_ROTATION_RANGE_RAD = (-np.pi / 6, np.pi / 6)  # x,y,z 回転の範囲（ラジアン）
    RANDOM_TRANSLATION_RANGE = (-0.1, 0.1)  # x,y,z 平行移動の範囲

    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        self.source = source
        self.target = target
        self.window_name = window_name
        self.settings: MatcherSettings | None = None
        self.is_logging = False
        self.last_ransac_result: o3d.pipelines.registration.RegistrationResult | None = None
        # 基準となるソース中心（sample.plyの重心）
        self.source_base_center = np.asarray(self.source.pcd.get_center())

        self.app = o3dv_gui.Application.instance
        self.app.initialize()
        self.view_manager = ViewManager(
            self.app,
            self.window_name,
            MatcherGeometyData(source=self.source, target=self.target),
        )

        # ボタンのコールバック設定
        self.view_manager.random_transform_button.set_on_clicked(self._on_random_transform)
        self.view_manager.ransac_button.set_on_clicked(self._on_run_ransac)
        self.view_manager.icp_button.set_on_clicked(self._on_run_icp)

    def invoke(self, settings: MatcherSettings, *, is_logging: bool) -> None:
        self.settings = settings
        self.is_logging = is_logging
        self.view_manager.window.post_redraw()

        # 毎ループmain threadから呼び出される処理
        self.view_manager.window.set_on_tick_event(self._on_tick)

        self.app.run()

    def _on_tick(self) -> bool:
        # キー入力など GUI 系の処理だけ行う。なければ単に False でもよい
        return False  # ここで True を返すと毎フレーム再描画要求になる

    def _on_run_ransac(self) -> None:
        """RANSACボタンがクリックされた時の処理"""
        if self.settings is None:
            logger.warning("Settings not initialized")
            return

        self.view_manager.label.text = "Running RANSAC..."
        self.view_manager.window.post_redraw()
        
        # RANSAC処理を別スレッドで実行
        self.app.run_in_thread(self._run_ransac_worker)

    def _on_run_icp(self) -> None:
        """ICPボタンがクリックされた時の処理"""
        if self.settings is None:
            logger.warning("Settings not initialized")
            return
        
        if self.last_ransac_result is None:
            self.view_manager.label.text = "Run RANSAC first!"
            self.view_manager.window.post_redraw()
            return

        self.view_manager.label.text = "Running ICP..."
        self.view_manager.window.post_redraw()
        
        # ICP処理を別スレッドで実行
        self.app.run_in_thread(self._run_icp_worker)

    def _on_random_transform(self) -> None:
        """ランダムな変換をソースポイントクラウドに適用"""
        # ランダムな回転行列を生成（オイラー角から）
        angles = np.random.uniform(
            self.RANDOM_ROTATION_RANGE_RAD[0],
            self.RANDOM_ROTATION_RANGE_RAD[1],
            3,
        )  # x, y, z軸周りのランダムな角度

        # 回転行列の生成
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]],
        )
        Ry = np.array(
            [[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]],
        )
        Rz = np.array(
            [[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]],
        )
        rotation = Rz @ Ry @ Rx

        # ランダムな平行移動ベクトル
        translation = np.random.uniform(
            self.RANDOM_TRANSLATION_RANGE[0],
            self.RANDOM_TRANSLATION_RANGE[1],
            3,
        )

        # 4x4変換行列を構築
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        # 基準中心を原点に移して回転し、再び戻したうえで平行移動を加える
        offset = -rotation @ self.source_base_center + self.source_base_center + translation
        transformation[:3, 3] = offset

        self._apply_transform_to_source(transformation, label="Random transformation applied")

    def _run_ransac_worker(self) -> None:
        """RANSACを実行するワーカースレッド"""
        if self.settings is None:
            return

        iter_num = 0
        result = None

        while iter_num < self.settings.ransac_iteration:
            result = global_registration(
                self.source,
                self.target,
                self.settings.voxel_size,
                iteration=1000,
            )
            iter_num += 1
            if self.is_logging:
                logger.info("RANSAC iteration %d/%d: %s", iter_num, self.settings.ransac_iteration, result)

            # main threadでgeometryを更新
            self.app.post_to_main_thread(self.view_manager.window, lambda res=result: self._apply_result(res))

        # 最後の結果を保存
        self.last_ransac_result = result

        # 完了メッセージを表示
        def update_label() -> None:
            self.view_manager.label.text = (
                f"RANSAC completed. Fitness: {result.fitness:.4f}" if result else "RANSAC failed"
            )
            self.view_manager.window.post_redraw()

        self.app.post_to_main_thread(self.view_manager.window, update_label)

    def _run_icp_worker(self) -> None:
        """ICPを実行するワーカースレッド"""
        if self.settings is None or self.last_ransac_result is None:
            return

        icp_result = refine_registration(
            self.source,
            self.target,
            self.last_ransac_result.transformation,
            self.settings.voxel_size,
        )

        if self.is_logging:
            logger.info("ICP result: %s", icp_result)

        # main threadでgeometryを更新
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._apply_result(icp_result))

        # 完了メッセージを表示
        def update_label() -> None:
            self.view_manager.label.text = f"ICP completed. Fitness: {icp_result.fitness:.4f}"
            self.view_manager.window.post_redraw()

        self.app.post_to_main_thread(self.view_manager.window, update_label)

    def _worker_loop(self, settings: MatcherSettings, *, is_logging: bool) -> None:
        iter_num = 0
        result = None

        while iter_num < settings.ransac_iteration:
            # ここは別スレッド → ICP/RANSAC 計算だけ
            result = global_registration(
                self.source,
                self.target,
                settings.voxel_size,
                iteration=1,
            )
            iter_num += 1
            if is_logging:
                logger.info("RANSAC iteration %d/%d: %s", iter_num, settings.ransac_iteration, result)

            # main thread で geometry を触るために post_to_main_thread
            self.app.post_to_main_thread(self.view_manager.window, lambda res=result: self._apply_result(res))

        # RANSAC 終了後に ICP 一回
        if result is not None:
            icp_result = refine_registration(self.source, self.target, result.transformation, settings.voxel_size)
            if is_logging:
                logger.info("ICP result: %s", icp_result)

            self.app.post_to_main_thread(self.view_manager.window, lambda res=icp_result: self._apply_result(res))

    def _apply_result(self, result: o3d.pipelines.registration.RegistrationResult) -> None:
        # ここは main thread 確定なので GUI 触ってよい
        self._apply_transform_to_source(result.transformation, label=f"Fitness: {result.fitness:.4f}")

    def _apply_transform_to_source(self, transformation: np.ndarray, *, label: str) -> None:
        """ソースの生点群とダウンサンプルを同期させて変換し、シーンを更新する"""
        self.source.pcd.transform(transformation)
        self.source.pcd_down.transform(transformation)

        if self.view_manager.scene.has_geometry(SOURCE_NAME):
            self.view_manager.scene.remove_geometry(SOURCE_NAME)
        self.view_manager.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.view_manager.material)

        self.view_manager.label.text = label
        self.view_manager.window.post_redraw()


if __name__ == "__main__":
    from pathlib import Path

    from ply import Ply

    voxel_size = 0.01
    base_path = Path(__file__).parent.parent.parent / "3d_data"
    src_path = base_path / "sample.ply"
    tgt_path = base_path / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    visualizer = VisualizeMatcher(src_ply, tgt_ply)
    visualizer.invoke(
        MatcherSettings(
            voxel_size=voxel_size,
            ransac_iteration=3,
        ),
        is_logging=True,
    )
