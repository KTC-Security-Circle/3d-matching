# ruff: noqa: INP001

from __future__ import annotations

import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app" / "core" / "src"))
sys.path.insert(0, str(ROOT / "app" / "serve" / "src"))

from serve.main import run  # noqa: E402


class Result:
    fitness = 0.9
    inlier_rmse = 0.1
    transformation = np.identity(4)
    correspondence_set = np.array([[0, 1]])


class ServeTests(unittest.TestCase):
    def run_protocol(self, input_text: str) -> list[dict[str, object]]:
        output = StringIO()
        with patch("serve.main.Ply"), patch("serve.main.global_registration") as ransac, patch(
            "serve.main.refine_registration", return_value=Result(),
        ):
            ransac.return_value = Result()
            run(StringIO(input_text), output)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_ready_and_valid_matching(self) -> None:
        responses = self.run_protocol(
            json.dumps(
                {
                    "command": "matching",
                    "source_path": "/data/source.ply",
                    "target_path": "/data/target.ply",
                    "voxel_size": 0.25,
                    "ransac_iterations": 30,
                },
            ),
        )
        assert responses[0] == {"ready": True}
        assert responses[1]["fitness"] == Result.fitness

    def test_invalid_input_and_matching_error_are_recoverable(self) -> None:
        responses = self.run_protocol("not json\n" + json.dumps({"command": "matching"}) + "\n")
        first_error = responses[1]["error"]
        second_error = responses[2]["error"]
        assert isinstance(first_error, dict)
        assert isinstance(second_error, dict)
        assert first_error["code"] == "invalid_json"
        assert second_error["code"] == "invalid_request"

    def test_shutdown_exits_without_response(self) -> None:
        assert self.run_protocol('{"command":"shutdown"}\n') == [{"ready": True}]
