from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        self.source = source
        self.target = target
        self.window_name = window_name

        self.app = o3dv_gui.Application.instance
        self.app.initialize()
        self.view_manager = ViewManager(
            self.app,
            self.window_name,
            ViewData(source=self.source.pcd, target=self.target.pcd),
        )

    def invoke(self, settings: MatcherSettings, *, is_logging: bool) -> None:
        self.view_manager.window.post_redraw()

        # 1. 毎ループmain threadから呼び出される処理
        self.view_manager.window.set_on_tick_event(self._on_tick)

        # 2. RANSAC/ICP は別スレッドに逃がす
        self.app.run_in_thread(lambda: self._worker_loop(settings, is_logging=is_logging))

        self.app.run()

    def _on_tick(self) -> bool:
        # キー入力など GUI 系の処理だけ行う。なければ単に False でもよい
        return False  # ここで True を返すと毎フレーム再描画要求になる

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
        self.source.pcd.transform(result.transformation)

        if self.view_manager.scene.has_geometry(SOURCE_NAME):
            self.view_manager.scene.remove_geometry(SOURCE_NAME)
        self.view_manager.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.view_manager.material)

        self.view_manager.label.text = f"Fitness: {result.fitness:.4f}"
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
