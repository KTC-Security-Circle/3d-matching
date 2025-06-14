from pathlib import Path

from matcher.icp import refine_registration
from matcher.ransac import global_registration
from ply import Ply
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)

DATA_DIRECTORY = (Path(__file__).parent / ".." / "3d_data").resolve()


def main() -> None:
    voxel_size = 0.01
    src_path = DATA_DIRECTORY / "sample.ply"
    tgt_path = DATA_DIRECTORY / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    init_trans = global_registration(src_ply, tgt_ply, voxel_size)  # RANSAC
    refine_registration(src_ply, tgt_ply, init_trans, voxel_size)  # ICP


if __name__ == "__main__":
    main()
