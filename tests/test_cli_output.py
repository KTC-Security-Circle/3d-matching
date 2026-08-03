from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app" / "core" / "src"))
sys.path.insert(0, str(ROOT / "app" / "cli" / "src"))

from cli.main import (  # noqa: E402
    OutputField,
    configure_output_mode,
    o3d,
    print_result,
    result_to_json,
)


class Result:
    fitness = 0.95
    inlier_rmse = 0.0123
    transformation = np.identity(4)
    correspondence_set = np.array([[1, 2], [3, 4]])


class CliOutputTests(unittest.TestCase):
    def test_result_to_json_preserves_all_registration_fields(self) -> None:
        result = result_to_json(Result())

        self.assertEqual(result["fitness"], 0.95)
        self.assertEqual(result["inlier_rmse"], 0.0123)
        self.assertEqual(result["transformation"], np.identity(4).tolist())
        self.assertEqual(result["correspondence_set"], [[1, 2], [3, 4]])

    def test_json_output_is_a_single_parseable_line(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            print_result(Result(), json_output=True, output=None)

        output = stdout.getvalue()
        self.assertEqual(output.count("\n"), 1)
        self.assertEqual(json.loads(output)["correspondence_set"], [[1, 2], [3, 4]])

    def test_json_rejects_output_field(self) -> None:
        with self.assertRaisesRegex(Exception, "--json and --output"):
            configure_output_mode(
                verbose=False,
                json_output=True,
                output=OutputField.FITNESS,
            )

    def test_json_rejects_verbose(self) -> None:
        with self.assertRaisesRegex(Exception, "--json and --verbose"):
            configure_output_mode(verbose=True, json_output=True, output=None)

    @patch("cli.main.configure_core_logging")
    @patch("cli.main.o3d.utility.set_verbosity_level")
    def test_verbose_configures_debug_logging(self, set_verbosity_level, configure_logging) -> None:
        configure_output_mode(verbose=True, json_output=False, output=None)

        set_verbosity_level.assert_called_once()
        configure_logging.assert_called_once_with(stream=sys.stdout, level=20)

    @patch("cli.main.configure_core_logging")
    @patch("cli.main.o3d.utility.set_verbosity_level")
    def test_normal_mode_resets_default_logging(self, set_verbosity_level, configure_logging) -> None:
        configure_output_mode(verbose=False, json_output=False, output=None)

        set_verbosity_level.assert_called_once_with(o3d.utility.VerbosityLevel.Info)
        configure_logging.assert_called_once_with(stream=sys.stderr, level=20)
