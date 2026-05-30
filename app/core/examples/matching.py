"""3D点群レジストレーション パイプラインのエントリーポイント.

処理の流れ:
    1. ソース/ターゲットのPLYファイルを読み込み・前処理(ダウンサンプル + FPFH特徴量計算)

    2. RANSACによるグローバルレジストレーション(粗い位置合わせ)

    3. ICPによるリファインメント(精密な位置合わせ)

    4. 各ステップの結果を3Dビューアで可視化
"""

from pathlib import Path

from core.matcher.icp import refine_registration
from core.matcher.ransac import global_registration
from core.ply import Ply
from core.utils.setup_logging import setup_logging
from core.visualization.draw_registration_result import draw_registration_result

logger = setup_logging(__name__)

# プロジェクトルート直下の 3d_data/ ディレクトリを参照
DATA_DIRECTORY = (Path(__file__).parent.parent.parent.parent / "3d_data").resolve()


def main() -> None:
    """点群レジストレーションの全工程を実行するメイン関数."""
    src_path = DATA_DIRECTORY / "sample.ply"
    tgt_path = DATA_DIRECTORY / "sample.ply"

    voxel_size = 0.3

    # PLYファイルの読み込みと前処理(ダウンサンプル、法線推定、FPFH特徴量計算)

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    # Step 1: RANSACによるグローバルレジストレーション(特徴量ベースの粗い位置合わせ)

    init_trans = global_registration(src_ply, tgt_ply, 0.3).transformation
    draw_registration_result(src_ply, tgt_ply, init_trans)

    # Step 2: ICPによるリファインメント(Point-to-Plane距離を用いた精密位置合わせ)

    icp_trains = refine_registration(src_ply, tgt_ply, init_trans, voxel_size).transformation
    draw_registration_result(src_ply, tgt_ply, icp_trains)


if __name__ == "__main__":
    main()
