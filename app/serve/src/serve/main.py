from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import open3d as o3d
from core.matcher.icp import refine_registration
from core.matcher.ransac import global_registration
from core.ply import Ply

if TYPE_CHECKING:
    from core.matcher.type import RegistrationResult


def _result_to_json(result: RegistrationResult) -> dict[str, object]:
    matrix_dimension = 2
    transformation = np.asarray(result.transformation, dtype=float)
    correspondence_set = np.asarray(result.correspondence_set, dtype=int)
    if transformation.shape != (4, 4):
        msg = "transformation must be a 4x4 matrix"
        raise ValueError(msg)
    if correspondence_set.ndim != matrix_dimension or correspondence_set.shape[1] != matrix_dimension:
        msg = "correspondence_set must be an Nx2 matrix"
        raise ValueError(msg)
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "transformation": transformation.tolist(),
        "correspondence_set": correspondence_set.tolist(),
    }


def _error(code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message}}


def _validate_matching_request(
    request: object,
) -> tuple[Path, Path, float, int] | dict[str, object]:
    if not isinstance(request, dict):
        return _error("invalid_request", "request must be a JSON object")
    request_data: dict[str, object] = {
        key: value for key, value in request.items() if isinstance(key, str)
    }
    if request_data.get("command") != "matching":
        return _error("unsupported_request", "only matching requests are supported")

    source = request_data.get("source_path")
    target = request_data.get("target_path")
    voxel_size = request_data.get("voxel_size")
    iterations = request_data.get("ransac_iterations")
    error: dict[str, object] | None = None
    if not isinstance(source, str) or not Path(source).is_absolute():
        error = _error("invalid_request", "source_path must be an absolute path")
    elif not isinstance(target, str) or not Path(target).is_absolute():
        error = _error("invalid_request", "target_path must be an absolute path")
    elif isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
        error = _error("invalid_request", "voxel_size must be a positive number")
    elif isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        error = _error("invalid_request", "ransac_iterations must be a positive integer")
    if error is not None:
        return error
    if not isinstance(source, str) or not isinstance(target, str):
        return _error("invalid_request", "source_path and target_path must be strings")
    if not isinstance(voxel_size, (int, float)) or not isinstance(iterations, int):
        return _error("invalid_request", "matching parameters have invalid types")
    return Path(source), Path(target), float(voxel_size), iterations


def _matching(request: object) -> dict[str, object]:
    validated = _validate_matching_request(request)
    if isinstance(validated, dict):
        return validated
    source, target, voxel_size, iterations = validated
    source_ply = Ply(source, voxel_size)
    target_ply = Ply(target, voxel_size)
    ransac_result = global_registration(source_ply, target_ply, voxel_size, iterations)
    result = refine_registration(source_ply, target_ply, ransac_result.transformation, voxel_size)
    return _result_to_json(result)


def run(stdin: IO[str], stdout: IO[str]) -> None:
    """Run the ready-then-JSONL matching protocol."""
    stdout.write(json.dumps({"ready": True}) + "\n")
    stdout.flush()
    for line in stdin:
        try:
            request = json.loads(line)
            if isinstance(request, dict) and request.get("command") == "shutdown":
                return
            response = _matching(request)
        except json.JSONDecodeError:
            response = _error("invalid_json", "input is not valid JSON")
        except Exception as exc:  # noqa: BLE001 - a bad request must not kill the worker.
            response = _error("matching_error", str(exc))
            sys.stderr.write(f"matching request failed: {exc}\n")
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()


def main() -> None:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    run(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
