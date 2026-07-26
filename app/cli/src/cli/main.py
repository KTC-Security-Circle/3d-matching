from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer
from core.matcher.icp import refine_registration
from core.matcher.ransac import global_registration
from core.ply import Ply

app = typer.Typer()


@app.command()
def ransac(
    source_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ソース点群のPLYファイルパス",
        ),
    ],
    target_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ターゲット点群のPLYファイルパス",
        ),
    ],
    voxel_size: Annotated[
        float,
        typer.Option(min=0.0, help="ダウンサンプリングのボクセルサイズ"),
    ] = 0.15,
    ransac_iterations: Annotated[
        int,
        typer.Option(min=1, help="RANSACのイテレーション回数"),
    ] = 30,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    print(rsc_result)


@app.command()
def icp(
    source_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ソース点群のPLYファイルパス",
        ),
    ],
    target_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ターゲット点群のPLYファイルパス",
        ),
    ],
    voxel_size: Annotated[
        float,
        typer.Option(min=0.0, help="ダウンサンプリングのボクセルサイズ"),
    ] = 0.25,
    ransac_iterations: Annotated[
        int,
        typer.Option(min=1, help="RANSACのイテレーション回数"),
    ] = 30,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print(icp_result)


@app.command()
def matching(
    source_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ソース点群のPLYファイルパス",
        ),
    ],
    target_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="ターゲット点群のPLYファイルパス",
        ),
    ],
    voxel_size: Annotated[
        float,
        typer.Option(min=0.0, help="ダウンサンプリングのボクセルサイズ"),
    ] = 0.25,
    ransac_iterations: Annotated[
        int,
        typer.Option(min=1, help="RANSACのイテレーション回数"),
    ] = 30,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print(icp_result)


if __name__ == "__main__":
    app()
