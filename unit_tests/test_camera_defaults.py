"""Full coverage tests for pyS3M.CameraDefaults -- camera pixel size / Bayer
mosaic unit / QE file registry lookup.
"""
import pytest

import pyS3M.CameraDefaults as CameraDefaults


class TestGetCameraConfig:
    def test_ximea(self):
        cfg = CameraDefaults.get_camera_config("ximea")
        assert cfg.pixel_size == pytest.approx(0.069)
        assert cfg.mosaic_unit.shape == (2, 2)

    def test_zwo(self):
        cfg = CameraDefaults.get_camera_config("zwo")
        assert cfg.pixel_size == pytest.approx(0.0715)

    def test_case_insensitive(self):
        cfg = CameraDefaults.get_camera_config("XIMEA")
        assert cfg.pixel_size == pytest.approx(0.069)

    def test_unknown_camera_raises(self):
        with pytest.raises(ValueError, match="Unknown camera"):
            CameraDefaults.get_camera_config("not_a_real_camera")
