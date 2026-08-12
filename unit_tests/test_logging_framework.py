"""Full coverage tests for pyS3M.LoggingFramework -- the custom formatter,
the file/console logger factory (with its module-level logger cache), and
the matplotlib log-level configuration helper.

Real `logging` records/handlers throughout (this is a thin wrapper over the
stdlib `logging` module -- no need to mock it). `setup_logger`'s per-name
cache is module-global state, so every test uses a unique logger name to
stay independent of test execution order (same concern as the shared-RNG
issue documented for FRCFunctions.py elsewhere in this coverage push).
"""
from __future__ import annotations

import logging
import sys

import pytest

import pyS3M.LoggingFramework as LoggingFramework


def _record(level=logging.INFO, msg="hello"):
    return logging.LogRecord(
        name="test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None, func="some_func",
    )


class TestPyBayerSMLMFormatter:
    def test_format_info_level_no_funcname(self):
        formatter = LoggingFramework.PyBayerSMLMFormatter()
        out = formatter.format(_record(level=logging.INFO))
        assert "hello" in out
        assert "some_func" not in out

    def test_format_debug_level_includes_funcname(self):
        formatter = LoggingFramework.PyBayerSMLMFormatter()
        out = formatter.format(_record(level=logging.DEBUG))
        assert "some_func" in out

    def test_format_includes_memory_when_psutil_available(self):
        formatter = LoggingFramework.PyBayerSMLMFormatter()
        out = formatter.format(_record())
        assert "MEM:" in out
        assert "N/A" not in out

    def test_format_falls_back_to_na_when_psutil_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        formatter = LoggingFramework.PyBayerSMLMFormatter()
        out = formatter.format(_record())
        assert "MEM:N/A" in out


class TestSetupLogger:
    def test_returns_logging_logger_with_correct_level(self):
        logger = LoggingFramework.setup_logger(
            "test_logger_basic", level=logging.WARNING, log_to_file=False
        )
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.WARNING
        assert logger.propagate is False

    def test_cached_on_second_call_with_same_name(self):
        logger1 = LoggingFramework.setup_logger("test_logger_cache", log_to_file=False)
        logger2 = LoggingFramework.setup_logger("test_logger_cache", log_to_file=False)
        assert logger1 is logger2

    def test_console_output_true_adds_stream_handler(self):
        logger = LoggingFramework.setup_logger(
            "test_logger_console_on", log_to_file=False, console_output=True
        )
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_console_output_false_adds_no_stream_handler(self):
        logger = LoggingFramework.setup_logger(
            "test_logger_console_off", log_to_file=False, console_output=False
        )
        assert not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )

    def test_log_to_file_with_explicit_dir_creates_file_handler(self, tmp_path):
        logger = LoggingFramework.setup_logger(
            "test_logger_file", log_to_file=True, log_dir=tmp_path, console_output=False
        )
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        log_files = list(tmp_path.glob("test_logger_file_*.log"))
        assert len(log_files) == 1

    def test_log_to_file_false_adds_no_file_handler(self, tmp_path):
        logger = LoggingFramework.setup_logger(
            "test_logger_nofile", log_to_file=False, log_dir=tmp_path, console_output=False
        )
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_log_dir_none_defaults_to_project_root_logs(self, tmp_path, monkeypatch):
        fake_module_path = tmp_path / "fakerepo" / "src" / "LoggingFramework.py"
        fake_module_path.parent.mkdir(parents=True)
        monkeypatch.setattr(LoggingFramework, "__file__", str(fake_module_path))
        logger = LoggingFramework.setup_logger(
            "test_logger_default_dir", log_to_file=True, log_dir=None, console_output=False
        )
        expected_dir = tmp_path / "fakerepo" / "logs"
        assert expected_dir.is_dir()
        log_files = list(expected_dir.glob("test_logger_default_dir_*.log"))
        assert len(log_files) == 1


class TestConfigureMatplotlibLogging:
    def test_sets_matplotlib_logger_to_warning(self):
        logging.getLogger("matplotlib").setLevel(logging.DEBUG)
        LoggingFramework.configure_matplotlib_logging()
        assert logging.getLogger("matplotlib").level == logging.WARNING
