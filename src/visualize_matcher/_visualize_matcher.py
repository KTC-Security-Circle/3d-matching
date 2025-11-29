from __future__ import annotations

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


class VisualizeMatcher:
    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        self.source = source
        self.target = target
        self.window_name = window_name

        self.app = o3dv_gui.Application.instance
        self.app.initialize()

        self.iter_num = 0
        self.max_iter = 0
        self.voxel_size = 0
        self.is_logging = False
        self._result = None
        self._is_executed_icp = False

    def _setup_app(self) -> None:
        self.window = self.app.create_window(self.window_name, 800, 600)

        # Scene Setup
        self.scene = o3dv_rendering.Open3DScene(self.window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, self.target.pcd, self.material)

        layout = o3dv_gui.Vert(0, o3dv_gui.Margins(10, 10, 10, 10))
        layout.background_color = o3dv_gui.Color(0, 0, 0, 1)

        self.label = o3dv_gui.Label("RANSAC Fitness: progressing...")

        scene_widget = o3dv_gui.SceneWidget()
        scene_widget.scene = self.scene
        scene_widget.setup_camera(60.0, self.scene.bounding_box, self.scene.bounding_box.get_center())

        layout.add_child(self.label)
        layout.add_child(scene_widget)

        self.window.add_child(layout)
        self.scene_widget = scene_widget

    def invoke(self, voxel_size: float, ransac_iteration: int, *, is_logging: bool) -> None:
        self.voxel_size = voxel_size
        self.max_iter = ransac_iteration
        self.is_logging = is_logging

        self._setup_app()
        self.window.set_on_tick_event(lambda: self._update_step())
        self.app.run()

    def _update_step(self) -> bool:
        """Called every frame by GUI."""
        if self.iter_num >= self.max_iter and self._is_executed_icp:
            return False

        if self.iter_num < self.max_iter:
            self._result = global_registration(
                self.source,
                self.target,
                self.voxel_size,
                iteration=1,
            )
        elif self._result is not None:
            self._result = refine_registration(
                self.source,
                self.target,
                self._result.transformation,
                self.voxel_size,
            )
            self._is_executed_icp = True

        def _task() -> None:
            if self._result is not None:
                self.label.text = f"Fitness: {self._result.fitness:.4f}"
                self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)

        # Update source geometry (your Ply.pcd is updated inside global_registration)
        self.app.post_to_main_thread(self.window, _task)

        self.iter_num += 1
        return True  # continue ticking


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
    visualizer.invoke(voxel_size, ransac_iteration=3, is_logging=True)
