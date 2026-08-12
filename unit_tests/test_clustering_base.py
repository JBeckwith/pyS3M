"""Full coverage tests for pyS3M.clustering._base.ClusteringBaseMixin's
`_check_min_locs` guard (the two early-return-False branches aren't exercised
by any real clustering pipeline's happy-path tests, which never run out of
localisations).
"""
import logging

import pandas as pd
import pytest

from pyS3M.clustering._base import ClusteringBaseMixin


class _Host(ClusteringBaseMixin):
    pass


@pytest.fixture
def host():
    return _Host()


class TestCheckMinLocs:
    def test_empty_dataframe_returns_false(self, host, caplog):
        with caplog.at_level(logging.WARNING):
            ok = host._check_min_locs(pd.DataFrame({"xc": []}), min_count=5)
        assert ok is False
        assert "No localizations remaining" in caplog.text

    def test_below_min_count_returns_false(self, host, caplog):
        with caplog.at_level(logging.WARNING):
            ok = host._check_min_locs(pd.DataFrame({"xc": [1.0, 2.0]}), min_count=5)
        assert ok is False
        assert "Only 2 localizations remaining" in caplog.text

    def test_sufficient_locs_returns_true(self, host):
        ok = host._check_min_locs(pd.DataFrame({"xc": [1.0, 2.0, 3.0]}), min_count=3)
        assert ok is True
