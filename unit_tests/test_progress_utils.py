"""Full coverage tests for pyS3M.ProgressUtils -- the tqdm-based
`clean_progress_bar` context manager (real tqdm + the no-tqdm mock fallback),
`ProgressBarManager`'s global-disable flag (both direct and env-var-driven),
and the `fitting_progress_bar`/`analysis_progress_bar` convenience wrappers.

`progress_manager` is a module-level singleton shared with the rest of the
package, so every test that touches its `_global_disable` flag or the
`PYBAYERSMLM_NO_PROGRESS` env var restores it in a `finally` block to avoid
leaking state into other tests.
"""
from __future__ import annotations

import importlib
import sys

import pytest

import pyS3M.ProgressUtils as ProgressUtils


class TestMockProgressUtils:
    def test_with_iterable(self):
        ctx = ProgressUtils.MockProgressUtils.clean_progress_bar(iterable=[1, 2, 3])
        with ctx as pbar:
            assert list(pbar) == [1, 2, 3]

    def test_with_total_only(self):
        ctx = ProgressUtils.MockProgressUtils.clean_progress_bar(total=5)
        with ctx as pbar:
            assert list(pbar) == list(range(5))

    def test_with_neither_iterable_nor_total(self):
        ctx = ProgressUtils.MockProgressUtils.clean_progress_bar()
        with ctx as pbar:
            assert list(pbar) == []

    def test_update_write_close_are_noop_safe(self, capsys):
        ctx = ProgressUtils.MockProgressUtils.clean_progress_bar(total=3)
        with ctx as pbar:
            ctx.update(1)
            ctx.write("hello")
            ctx.close()
        out = capsys.readouterr().out
        assert "hello" in out


class TestProgressBarConfig:
    def test_get_default_kwargs_contains_expected_keys(self):
        kwargs = ProgressUtils.ProgressBarConfig.get_default_kwargs()
        assert kwargs["leave"] is False
        assert kwargs["dynamic_ncols"] is True
        assert kwargs["file"] is not None


class TestTqdmImportFallback:
    def test_import_error_sets_base_tqdm_none_and_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tqdm", None)
        try:
            importlib.reload(ProgressUtils)
            assert ProgressUtils.base_tqdm is None
            assert ProgressUtils.TQDM_AVAILABLE is False
        finally:
            monkeypatch.undo()
            importlib.reload(ProgressUtils)


class TestCleanProgressBarMockFallback:
    def test_uses_mock_when_tqdm_unavailable(self, monkeypatch):
        # clean_progress_bar yields the raw (un-entered) MockContext object
        # directly when TQDM_AVAILABLE is False -- it exposes .iterable itself,
        # rather than needing __enter__ (matching MockProgressUtils's own
        # __enter__ semantics test above).
        monkeypatch.setattr(ProgressUtils, "TQDM_AVAILABLE", False)
        with ProgressUtils.clean_progress_bar(iterable=[1, 2, 3], desc="x") as pbar:
            assert pbar.iterable == [1, 2, 3]


class TestCleanProgressBarRealTqdm:
    def test_iterable_branch(self):
        seen = []
        with ProgressUtils.clean_progress_bar(iterable=[1, 2, 3], desc="items") as pbar:
            for item in pbar:
                seen.append(item)
        assert seen == [1, 2, 3]

    def test_total_branch_manual_update(self):
        with ProgressUtils.clean_progress_bar(total=5, desc="manual") as pbar:
            for _ in range(5):
                pbar.update(1)
        assert pbar.n == 5

    def test_neither_iterable_nor_total_raises(self):
        with pytest.raises(ValueError, match="Either 'iterable' or 'total'"):
            with ProgressUtils.clean_progress_bar():
                pass

    def test_leave_colour_position_overrides(self):
        with ProgressUtils.clean_progress_bar(
            total=2, leave=True, colour="red", position=0
        ) as pbar:
            pbar.update(2)
        assert pbar.n == 2

    def test_globally_disabled_sets_disable_kwarg(self):
        ProgressUtils.set_progress_enabled(False)
        try:
            with ProgressUtils.clean_progress_bar(total=3) as pbar:
                pbar.update(3)
            assert pbar.disable is True
        finally:
            ProgressUtils.set_progress_enabled(True)

    def test_keyboard_interrupt_propagates_and_writes(self, capsys):
        with pytest.raises(KeyboardInterrupt):
            with ProgressUtils.clean_progress_bar(total=3, desc="interrupt-me") as pbar:
                raise KeyboardInterrupt()
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_generic_exception_propagates_and_writes(self, capsys):
        with pytest.raises(RuntimeError, match="boom"):
            with ProgressUtils.clean_progress_bar(total=3, desc="fail-me") as pbar:
                raise RuntimeError("boom")
        out = capsys.readouterr().out
        assert "Error during operation" in out


class TestProgressBarManager:
    def test_set_global_disable_direct_flag(self):
        mgr = ProgressUtils.ProgressBarManager()
        assert mgr.is_globally_disabled() is False
        mgr.set_global_disable(True)
        assert mgr.is_globally_disabled() is True

    def test_env_var_disables_regardless_of_flag(self, monkeypatch):
        mgr = ProgressUtils.ProgressBarManager()
        monkeypatch.setenv("PYBAYERSMLM_NO_PROGRESS", "true")
        assert mgr.is_globally_disabled() is True

    def test_env_var_false_values_do_not_disable(self, monkeypatch):
        mgr = ProgressUtils.ProgressBarManager()
        monkeypatch.setenv("PYBAYERSMLM_NO_PROGRESS", "0")
        assert mgr.is_globally_disabled() is False


class TestSetProgressEnabled:
    def test_false_then_true_roundtrip(self):
        ProgressUtils.set_progress_enabled(False)
        try:
            assert ProgressUtils.progress_manager.is_globally_disabled() is True
        finally:
            ProgressUtils.set_progress_enabled(True)
        assert ProgressUtils.progress_manager.is_globally_disabled() is False


class TestConvenienceWrappers:
    def test_fitting_progress_bar_defaults(self):
        with ProgressUtils.fitting_progress_bar(total=2) as pbar:
            pbar.update(2)
        assert pbar.desc == "Fitting puncta"

    def test_analysis_progress_bar_defaults(self):
        with ProgressUtils.analysis_progress_bar(total=2) as pbar:
            pbar.update(2)
        assert pbar.desc == "Analysing data"
