"""Full coverage tests for pyS3M.ImportManager -- registered/lazy module
lookup, availability checks, and register_module's success/ImportError/
generic-Exception branches (with and without a required flag or fallback).

Real imports/`ImportError`s throughout (this module *is* an import-wrapper,
so mocking `__import__` would just test the mock). Each `ImportManager()` is
constructed fresh per test to avoid interference from the shared module-level
singleton (`_import_manager`) other tests/modules may have already populated.
"""
from __future__ import annotations

import sys
import warnings

import pytest

import pyS3M.ImportManager as ImportManager

ImportStatus = ImportManager.ImportStatus


class TestSetupPath:
    def test_adds_src_dir_to_syspath_when_missing(self, monkeypatch):
        from pathlib import Path
        src_dir = str(Path(ImportManager.__file__).parent)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != src_dir])
        assert src_dir not in sys.path
        ImportManager.ImportManager()
        assert src_dir in sys.path

    def test_does_not_duplicate_if_already_present(self, monkeypatch):
        from pathlib import Path
        src_dir = str(Path(ImportManager.__file__).parent)
        monkeypatch.setattr(sys, "path", list(sys.path) + [src_dir] if src_dir not in sys.path else list(sys.path))
        before = sys.path.count(src_dir)
        ImportManager.ImportManager()
        assert sys.path.count(src_dir) == before


class TestRegisterModule:
    def test_successful_import_with_alias(self):
        mgr = ImportManager.ImportManager()
        info = mgr.register_module("json", alias="json_alias", required=False)
        assert info.status == ImportStatus.AVAILABLE
        assert info.module is not None

    def test_successful_import_without_alias(self):
        mgr = ImportManager.ImportManager()
        info = mgr.register_module("json", required=False)
        assert info.status == ImportStatus.AVAILABLE

    def test_custom_import_used_when_provided(self):
        mgr = ImportManager.ImportManager()
        sentinel = object()
        info = mgr.register_module("fake_module", custom_import=lambda: sentinel)
        assert info.status == ImportStatus.AVAILABLE
        assert info.module is sentinel

    def test_import_error_required_module_warns_and_sets_none(self):
        mgr = ImportManager.ImportManager()
        with pytest.warns(UserWarning, match="CRITICAL"):
            info = mgr.register_module("no_such_module_xyz", required=True)
        assert info.status == ImportStatus.MISSING
        assert info.module is None
        assert info.required is True

    def test_import_error_optional_module_uses_fallback(self):
        mgr = ImportManager.ImportManager()
        fallback = object()
        with pytest.warns(UserWarning, match="Optional dependency missing"):
            info = mgr.register_module(
                "no_such_module_xyz2", required=False, fallback=fallback
            )
        assert info.status == ImportStatus.MISSING
        assert info.module is fallback

    def test_generic_exception_during_custom_import_sets_error_status(self):
        mgr = ImportManager.ImportManager()

        def _raiser():
            raise RuntimeError("boom")

        fallback = object()
        with pytest.warns(UserWarning, match="Error importing"):
            info = mgr.register_module(
                "broken_module", custom_import=_raiser, fallback=fallback
            )
        assert info.status == ImportStatus.ERROR
        assert info.module is fallback
        assert "boom" in info.error_message

    def test_registered_module_stored_in_modules_dict(self):
        mgr = ImportManager.ImportManager()
        mgr.register_module("json", required=False)
        assert "json" in mgr.modules


class TestGetModule:
    def test_returns_already_registered_module(self):
        mgr = ImportManager.ImportManager()
        mgr.register_module("json", required=False)
        assert mgr.get_module("json") is not None

    def test_lazy_loads_unregistered_module(self):
        mgr = ImportManager.ImportManager()
        assert "csv" not in mgr.modules
        result = mgr.get_module("csv")
        assert result is not None
        assert "csv" in mgr.modules
        assert mgr.modules["csv"].status == ImportStatus.AVAILABLE

    def test_lazy_load_import_error_returns_none(self):
        mgr = ImportManager.ImportManager()
        with pytest.warns(UserWarning, match="could not be imported"):
            result = mgr.get_module("no_such_module_lazy_xyz")
        assert result is None


class TestIsAvailable:
    def test_unregistered_module_returns_false(self):
        mgr = ImportManager.ImportManager()
        assert mgr.is_available("totally_unregistered_module") is False

    def test_registered_available_module_returns_true(self):
        mgr = ImportManager.ImportManager()
        mgr.register_module("json", required=False)
        assert mgr.is_available("json") is True

    def test_registered_missing_module_returns_false(self):
        mgr = ImportManager.ImportManager()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mgr.register_module("no_such_module_avail_xyz", required=False)
        assert mgr.is_available("no_such_module_avail_xyz") is False


class TestModuleLevelWrappers:
    def test_get_module_wrapper_delegates_to_singleton(self):
        result = ImportManager.get_module("numpy")
        assert result is not None

    def test_is_available_wrapper_delegates_to_singleton(self):
        assert ImportManager.is_available("numpy") is True
        assert ImportManager.is_available("totally_unregistered_module_2") is False
