"""RANSAC & ICP のインタラクティブ可視化モジュール。

Open3DのGUIフレームワークを使用して、レジストレーションアルゴリズムの
ステップバイステップ実行をリアルタイムに可視化する。

主なコンポーネント:
    - MatcherGeometyData: ソース/ターゲット点群を保持するデータクラス
    - MatcherSettings: レジストレーションパラメータの設定
    - ViewManager: Open3D GUIウィンドウの構築と管理
    - VisualizeMatcher: レジストレーション処理のオーケストレーション

GUIの機能:
    - "Random Transform": ソース点群にランダムな回転+平行移動を適用
    - "Run RANSAC (Manual Step)": RANSACを1イテレーションずつ可視化しながら実行
    - "Run ICP": RANSAC結果を初期値としてICP精密化を実行
"""

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
    compute_step_transformation,
    evaluate_inlier_ratio,
)
from utils.setup_logging import setup_logging

if TYPE_CHECKING:
    from ply import Ply

logger = setup_logging(__name__)

# 3Dシーン上でのジオメトリ識別名
SOURCE_NAME = "source"
TARGET_NAME = "target"


@dataclass
class MatcherGeometyData:
    """ソースとターゲットの点群ペアを保持するデータクラス。"""

    source: Ply
    target: Ply


class ViewManager:
    """Open3D GUIウィンドウの構築と管理を担当するクラス。

    3Dシーンの描画、UIコンポーネント（ボタン、ラベル）の配置、
    カメラ設定などを行う。

    Attributes:
        window: Open3DのGUIウィンドウ
        scene: 3D描画シーン
        material: 点群の描画マテリアル（点サイズなど）
        label: ステータス表示用ラベル
        info_label: フィットネス値表示用ラベル
        random_transform_button: ランダム変換ボタン
        ransac_button: RANSAC実行ボタン
        icp_button: ICP実行ボタン
    """

    def __init__(self, app: o3dv_gui.Application, window_name: str, init_data: MatcherGeometyData) -> None:
        """GUIウィンドウを作成し、3Dシーンとコントロールパネルを構築する。

        Args:
            app: Open3Dアプリケーションインスタンス
            window_name: ウィンドウタイトル
            init_data: 初期表示する点群データ
        """
        self.window = app.create_window(window_name, 1024, 768)

        # 3Dシーンとマテリアルの設定
        self.scene = o3dv_rendering.Open3DScene(self.window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.material.point_size = 5.0

        # ソースとターゲットの点群を3Dシーンに追加
        self.scene.add_geometry(SOURCE_NAME, init_data.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, init_data.target.pcd, self.material)

        # --- GUIレイアウトの構築（左側コントロールパネル） ---
        em = self.window.theme.font_size
        gui_layout = o3dv_gui.Vert(0, o3dv_gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        gui_layout.frame = o3dv_gui.Rect(
            self.window.content_rect.x,
            self.window.content_rect.y,
            250,  # コントロールパネルの幅（ピクセル）
            self.window.content_rect.height,
        )

        # ステータスラベル（現在のイテレーション数など）
        self.label = o3dv_gui.Label("Ready")
        gui_layout.add_child(self.label)

        # フィットネス値の詳細表示ラベル
        self.info_label = o3dv_gui.Label("")
        gui_layout.add_child(self.info_label)

        # 操作ボタン群
        self.random_transform_button = o3dv_gui.Button("Random Transform")
        gui_layout.add_child(self.random_transform_button)

        self.ransac_button = o3dv_gui.Button("Run RANSAC (Manual Step)")
        gui_layout.add_child(self.ransac_button)

        self.icp_button = o3dv_gui.Button("Run ICP")
        gui_layout.add_child(self.icp_button)

        # --- 3Dシーンウィジェットの構築（右側メインビュー） ---
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
    """レジストレーションの動作パラメータ設定。

    Attributes:
        voxel_size: ボクセルサイズ。距離閾値やダウンサンプリングの基準
        ransac_iteration: RANSACの最大イテレーション数
        noise_ratio: ノイズ比率。偽対応点の混入割合（デフォルト: 2.0 = 元の2倍）
    """

    voxel_size: float
    ransac_iteration: int
    noise_ratio: float = 2.0


class VisualizeMatcher:
    """RANSAC & ICPレジストレーションのインタラクティブ可視化を管理するクラス。

    GUIボタン操作で以下を実行できる:
        1. ソース点群にランダムな剛体変換を適用（テスト用の初期位置ずれを生成）
        2. RANSACをステップバイステップで実行し、各イテレーションの結果をリアルタイム表示
        3. RANSAC結果を初期値としてICPによる精密化を実行

    Attributes:
        RANDOM_ROTATION_RANGE_RAD: ランダム回転の範囲（±30度 = ±π/6ラジアン）
        RANDOM_TRANSLATION_RANGE: ランダム平行移動の範囲（±0.1）
    """

    RANDOM_ROTATION_RANGE_RAD = (-np.pi / 6, np.pi / 6)
    RANDOM_TRANSLATION_RANGE = (-0.1, 0.1)

    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        """可視化ツールを初期化する。

        Args:
            source: ソース点群（変換対象）
            target: ターゲット点群（基準）
            window_name: GUIウィンドウのタイトル
        """
        self.source = source
        self.target = target
        self.window_name = window_name
        self.settings: MatcherSettings | None = None
        self.is_logging = False
        self.last_ransac_result = None

        # ソース点群の初期重心位置（ランダム変換時の回転中心として使用）
        self.source_base_center = np.asarray(self.source.pcd.get_center())
        self.rng = np.random.default_rng()

        # 変換前の点群を保持（各イテレーションでのプレビュー表示に使用）
        self.source_pcd_orig = copy.deepcopy(self.source.pcd)
        self.source_down_orig = copy.deepcopy(self.source.pcd_down)

        # Open3D GUIアプリケーションの初期化
        self.app = o3dv_gui.Application.instance
        self.app.initialize()
        self.view_manager = ViewManager(
            self.app,
            self.window_name,
            MatcherGeometyData(source=self.source, target=self.target),
        )

        # ボタンにイベントハンドラを登録
        self.view_manager.random_transform_button.set_on_clicked(self._on_random_transform)
        self.view_manager.ransac_button.set_on_clicked(self._on_run_ransac)
        self.view_manager.icp_button.set_on_clicked(self._on_run_icp)

    def invoke(self, settings: MatcherSettings, *, is_logging: bool) -> None:
        """可視化ツールを起動する。

        GUIウィンドウを表示し、ユーザーの操作を待ち受けるイベントループに入る。

        Args:
            settings: レジストレーションのパラメータ設定
            is_logging: ログ出力の有効/無効
        """
        self.settings = settings
        self.is_logging = is_logging
        self.view_manager.window.post_redraw()
        self.app.run()

    # ================================
    # ボタンイベントハンドラ
    # ================================

    def _on_run_ransac(self) -> None:
        """RANSACボタン押下時のハンドラ。別スレッドでRANSACワーカーを起動する。"""
        if self.settings is None:
            return
        self.view_manager.label.text = "Initializing..."
        self.view_manager.window.post_redraw()
        # UIスレッドをブロックしないよう、別スレッドで実行
        self.app.run_in_thread(self._run_ransac_worker)

    def _on_run_icp(self) -> None:
        """ICPボタン押下時のハンドラ。RANSACが未実行の場合は警告を表示する。"""
        if self.settings is None or self.last_ransac_result is None:
            self.view_manager.label.text = "Run RANSAC first!"
            return
        self.view_manager.label.text = "Running ICP..."
        self.view_manager.window.post_redraw()
        self.app.run_in_thread(self._run_icp_worker)

    def _on_random_transform(self) -> None:
        """ソース点群にランダムな剛体変換を適用する。

        X/Y/Z各軸に±30度のランダム回転と±0.1のランダム平行移動を生成し、
        ソース点群の重心を回転中心として変換を適用する。
        レジストレーションアルゴリズムのテスト用に初期位置ずれを生成する。
        """
        # X/Y/Z各軸の回転角をランダム生成
        angles = self.rng.uniform(*self.RANDOM_ROTATION_RANGE_RAD, 3)

        # 各軸の回転行列を作成（オイラー角 → 回転行列）
        rx = np.array(
            [[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]],
        )
        ry = np.array(
            [[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]],
        )
        rz = np.array(
            [[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]],
        )
        # 合成回転行列: R = Rz @ Ry @ Rx（Z-Y-X オイラー角の順序）
        rotation = rz @ ry @ rx
        translation = self.rng.uniform(*self.RANDOM_TRANSLATION_RANGE, 3)

        # 4x4同次変換行列の構築
        # 重心を回転中心とするため、offset = -R @ center + center + t
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        offset = -rotation @ self.source_base_center + self.source_base_center + translation
        transformation[:3, 3] = offset

        # フル解像度・ダウンサンプル両方の点群に変換を適用
        self.source.pcd.transform(transformation)
        self.source.pcd_down.transform(transformation)

        # 変換後の状態を保存（RANSACの各イテレーション表示の基準となる）
        self.source_pcd_orig = copy.deepcopy(self.source.pcd)
        self.source_down_orig = copy.deepcopy(self.source.pcd_down)

        # 3Dシーンを更新
        self._update_scene(self.source.pcd)
        self.view_manager.label.text = "Random Transformed"
        self.view_manager.window.post_redraw()

    # ================================
    # ワーカースレッド（バックグラウンド処理）
    # ================================

    def _run_ransac_worker(self) -> None:
        """RANSACをステップバイステップで実行するワーカー関数（別スレッドで実行）。

        各イテレーションで:
            1. 対応点から3点をランダムサンプリング
            2. Kabschアルゴリズムで変換行列を推定
            3. インライア率を計算してベストを更新
            4. UIスレッドに描画更新を通知（30ms間隔）
        """
        if self.settings is None:
            return

        # FPFH特徴量ベースの対応点を計算（ノイズ混入あり）
        corres = compute_feature_correspondences(self.source, self.target, noise_ratio=self.settings.noise_ratio)

        iter_num = 0
        max_iter = self.settings.ransac_iteration

        best_result = None
        best_fitness = -1.0

        logger.info(f"Start RANSAC: {len(corres)} correspondences (Noise Ratio: {self.settings.noise_ratio})")

        while iter_num < max_iter:
            iter_num += 1

            # 対応点から3点をサンプリングしてKabschアルゴリズムで変換行列を推定
            result = compute_step_transformation(self.source, self.target, corres)

            # 推定した変換行列の品質をインライア率で評価
            w_current = evaluate_inlier_ratio(
                self.source,
                self.target,
                corres,
                result.transformation,
                self.settings.voxel_size,
            )
            result.fitness = w_current

            # ベストスコアの更新
            if best_result is None or w_current > best_fitness:
                best_result = result
                best_fitness = w_current

            # メインスレッド（UIスレッド）に描画更新をポスト
            self.app.post_to_main_thread(
                self.view_manager.window,
                lambda res=result, it=iter_num, val=w_current, b_fit=best_fitness: self._update_viz(
                    res,
                    it,
                    val,
                    b_fit,
                ),
            )

            # フレームレート調整（30ms = 約33fps）
            time.sleep(0.03)

        # 全イテレーション完了後、ベスト結果で最終変換を適用
        self.last_ransac_result = best_result
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._finalize_ransac(best_result))

    def _update_viz(self, result, iter_num, w, best_fit):
        """RANSACの各イテレーション結果を3Dシーンに反映する（UIスレッドで実行）。

        Args:
            result: 現在のイテレーションで推定した変換結果
            iter_num: 現在のイテレーション番号
            w: 現在のインライア率
            best_fit: これまでの最良インライア率
        """
        # 元の点群のコピーに対して現在の変換行列を適用してプレビュー表示
        temp = copy.deepcopy(self.source_pcd_orig)
        temp.transform(result.transformation)
        self._update_scene(temp)

        # ステータスラベルの更新
        self.view_manager.label.text = f"Iter: {iter_num}"
        self.view_manager.info_label.text = f"CurFit: {w:.4f} | BestFit: {best_fit:.4f}"
        self.view_manager.window.post_redraw()

    def _finalize_ransac(self, result):
        """RANSAC完了後の最終処理。ベスト変換行列をソース点群に恒久的に適用する。

        Args:
            result: RANSACのベスト結果（None の場合は失敗）
        """
        if result:
            # ベスト変換行列をソース点群に適用（以後のICP処理の起点となる）
            self.source.pcd.transform(result.transformation)
            self.source.pcd_down.transform(result.transformation)
            # 変換後の状態を保存
            self.source_pcd_orig = copy.deepcopy(self.source.pcd)
            self.source_down_orig = copy.deepcopy(self.source.pcd_down)
            self._update_scene(self.source.pcd)
            self.view_manager.label.text = f"Done. Final Best: {result.fitness:.4f}"
        else:
            self.view_manager.label.text = "Failed."
        self.view_manager.window.post_redraw()

    def _run_icp_worker(self) -> None:
        """ICPリファインメントを実行するワーカー関数（別スレッドで実行）。

        RANSAC後の状態を起点に、単位行列を初期変換としてICPを実行する。
        （RANSAC結果は既にソース点群に適用済みのため、追加の変換を求める）
        """
        if self.settings is None or self.last_ransac_result is None:
            return
        # 初期変換は単位行列: RANSAC結果は既にpcdに適用済みなので追加の差分を求める
        res = refine_registration(self.source, self.target, np.eye(4), self.settings.voxel_size)
        self.app.post_to_main_thread(self.view_manager.window, lambda: self._finalize_icp(res))

    def _finalize_icp(self, result):
        """ICP完了後の最終処理。ICP結果の変換行列をソース点群に適用する。

        Args:
            result: ICPの結果（変換行列とフィットネス値）
        """
        self.source.pcd.transform(result.transformation)
        self.source.pcd_down.transform(result.transformation)
        self._update_scene(self.source.pcd)
        self.view_manager.label.text = f"ICP Done. Fit: {result.fitness:.4f}"
        self.view_manager.window.post_redraw()

    # ================================
    # 3Dシーン更新ユーティリティ
    # ================================

    def _update_scene(self, pcd):
        """3Dシーン上のソース点群ジオメトリを更新する。

        既存のソースジオメトリを削除し、新しい点群を追加する。

        Args:
            pcd: 表示する点群オブジェクト
        """
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
