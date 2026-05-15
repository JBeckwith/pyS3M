from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class _FigureTab(QWidget):
    """Tab that shows a placeholder until a figure is provided."""

    def __init__(self, placeholder_msg: str, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(placeholder_msg)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; font-size: 13pt;")
        self._lay.addWidget(self._placeholder)
        self._canvas = None
        self._toolbar = None

    def set_figure(self, fig: Figure):
        if self._canvas is not None:
            self._lay.removeWidget(self._toolbar)
            self._lay.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()
        self._placeholder.setVisible(False)
        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._lay.addWidget(self._toolbar)
        self._lay.addWidget(self._canvas)
        self._canvas.draw()


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        lay.addWidget(self._tabs)

        self._preview_tab = _FigureTab("Run fitting to see a preview.")
        self._tabs.addTab(self._preview_tab, "Preview")

        self._locs_tab = _FigureTab("Run Filter & Cluster to see results.")
        self._tabs.addTab(self._locs_tab, "Localisations")

    def set_preview_figure(self, fig: Figure):
        self._preview_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._preview_tab)

    def set_localisations_figure(self, fig: Figure):
        self._locs_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._locs_tab)
