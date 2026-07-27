from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import typer
from core.matcher.icp import refine_registration
from core.matcher.ransac import global_registration
from core.ply import Ply

if TYPE_CHECKING:
    from core.matcher.type import RegistrationResult

app = typer.Typer()


class OutputField(StrEnum):
    """CLIに出力するRegistrationResultのフィールド選択肢."""

    FITNESS = "fitness"
    INLIER_RMSE = "inlier_rmse"
    TRANSFORMATION = "transformation"
    CORRESPONDENCE_SET = "correspondence_set"


def format_result(
    result: RegistrationResult,
    output: OutputField | None,
) -> str:
    """RegistrationResultをCLI出力用にフォーマットする.

    Args:
        result: 出力対象のRegistrationResult。
        output: 出力するフィールド。None の場合は結果全体を文字列化して返す。

    Returns:
        指定されたフィールドの値、または結果全体の文字列表現。
    """
    if output is None:
        return str(result)
    return str(getattr(result, output.value))


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
    output: Annotated[
        OutputField | None,
        typer.Option(
            "--output",
            "-o",
            case_sensitive=False,
            help="CLIに出力するRegistrationResultのフィールド。未指定時は結果全体をprintする。",
        ),
    ] = None,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    print(format_result(rsc_result, output))


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
    output: Annotated[
        OutputField | None,
        typer.Option(
            "--output",
            "-o",
            case_sensitive=False,
            help="CLIに出力するRegistrationResultのフィールド。未指定時は結果全体をprintする。",
        ),
    ] = None,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print(format_result(icp_result, output))


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
    output: Annotated[
        OutputField | None,
        typer.Option(
            "--output",
            "-o",
            case_sensitive=False,
            help="CLIに出力するRegistrationResultのフィールド。未指定時は結果全体をprintする。",
        ),
    ] = None,
) -> None:
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print(format_result(icp_result, output))


if __name__ == "__main__":
    app()
