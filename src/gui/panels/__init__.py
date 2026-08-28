"""
panels — MainWindow's dock widgets, one per pipeline stage/tab (calibration,
setup, fitting, post-processing/clustering, drift, FRC, channel unmixing,
Nile Red, simulation, results). No package-level re-exports here — each
panel is imported directly by module, e.g. ``from pyS3M.gui.panels.frc_panel
import FRCPanel``.
"""
