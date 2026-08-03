from __future__ import annotations

import logging
import sys
import unittest
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app" / "core" / "src"))

from core.utils.setup_logging import configure_core_logging, setup_logging  # noqa: E402


class SetupLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger_name = f"core.tests.setup_logging.{self.id()}"
        self.logger = logging.getLogger(self.logger_name)
        self.logger.handlers.clear()

    def tearDown(self) -> None:
        self.logger.handlers.clear()

    def test_default_handler_writes_to_stderr(self) -> None:
        logger = setup_logging(self.logger_name)

        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        self.assertIs(logger.handlers[0].stream, sys.stderr)

    def test_setup_does_not_duplicate_handlers(self) -> None:
        setup_logging(self.logger_name)
        logger = setup_logging(self.logger_name)

        self.assertEqual(len(logger.handlers), 1)

    def test_cli_configuration_switches_existing_logger_to_stdout(self) -> None:
        logger = setup_logging(self.logger_name)

        configure_core_logging(stream=sys.stdout)

        self.assertIs(logger.handlers[0].stream, sys.stdout)

    def test_custom_stream_receives_logs(self) -> None:
        stream = StringIO()
        logger = setup_logging(self.logger_name, stream=stream)

        logger.info("configured")

        self.assertIn("configured", stream.getvalue())
