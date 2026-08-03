from __future__ import annotations

import json
import logging
import sys
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import numpy as np
import open3d as o3d
import typer
from core.matcher.icp import refine_registration
from core.matcher.ransac import global_registration
from core.ply import Ply
from core.utils.setup_logging import configure_core_logging

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


def result_to_json(result: RegistrationResult) -> dict[str, object]:
    """RegistrationResult を JSON と互換なプリミティブ値に変換する."""
    transformation = np.asarray(result.transformation, dtype=float)
    correspondence_set = np.asarray(result.correspondence_set, dtype=int)
    if transformation.shape != (4, 4):
        msg = "transformation must be a 4x4 matrix"
        raise ValueError(msg)
    if correspondence_set.ndim != 2 or correspondence_set.shape[1] != 2:
        msg = "correspondence_set must be an Nx2 matrix"
        raise ValueError(msg)
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "transformation": transformation.tolist(),
        "correspondence_set": correspondence_set.tolist(),
    }


def configure_output_mode(*, verbose: bool, json_output: bool, output: OutputField | None) -> None:
    """CLI のログ出力先と Open3D verbosity をオプションに従って設定する."""
    if json_output and verbose:
        raise typer.BadParameter("--json and --verbose cannot be used together")
    if json_output and output is not None:
        raise typer.BadParameter("--json and --output cannot be used together")

    if json_output:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        configure_core_logging(stream=sys.stderr, level=logging.INFO)
    elif verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        configure_core_logging(stream=sys.stdout, level=logging.INFO)
    else:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        configure_core_logging(stream=sys.stderr, level=logging.INFO)


def print_result(
    result: RegistrationResult,
    *,
    json_output: bool,
    output: OutputField | None,
) -> None:
    """モードに応じて RegistrationResult を標準出力へ出力する."""
    if json_output:
        print(json.dumps(result_to_json(result)))
        return
    print(format_result(result, output))


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
    verbose: Annotated[
        bool,
        typer.Option(help="core と Open3D の詳細ログを stdout に出力する。"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="結果を JSON 1行で stdout に出力する。"),
    ] = False,
) -> None:
    configure_output_mode(verbose=verbose, json_output=json_output, output=output)
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    print_result(rsc_result, json_output=json_output, output=output)


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
    verbose: Annotated[
        bool,
        typer.Option(help="core と Open3D の詳細ログを stdout に出力する。"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="結果を JSON 1行で stdout に出力する。"),
    ] = False,
) -> None:
    configure_output_mode(verbose=verbose, json_output=json_output, output=output)
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print_result(icp_result, json_output=json_output, output=output)


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
    verbose: Annotated[
        bool,
        typer.Option(help="core と Open3D の詳細ログを stdout に出力する。"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="結果を JSON 1行で stdout に出力する。"),
    ] = False,
) -> None:
    configure_output_mode(verbose=verbose, json_output=json_output, output=output)
    source_ply = Ply(source_path, voxel_size)
    target_ply = Ply(target_path, voxel_size)

    rsc_result = global_registration(source_ply, target_ply, voxel_size, ransac_iterations)
    icp_result = refine_registration(source_ply, target_ply, rsc_result.transformation, voxel_size)
    print_result(icp_result, json_output=json_output, output=output)


if __name__ == "__main__":
    app()
