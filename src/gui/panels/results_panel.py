from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class _DraggableZoomRect:
    """Moves paired Rectangle patches across two full-field axes in sync and
    updates a matching pair of zoom axes live.  Must be instantiated after the
    FigureCanvasQTAgg is created so mpl_connect targets the live Qt canvas."""

    def __init__(self, rects, zoom_axes, image_shape):
        self._rects = rects
        self._zoom_axes = zoom_axes
        self._h, self._w = image_shape[:2]
        self._press = None

        canvas = rects[0].figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _clamp(self, x0, y0, w, h):
        return max(0, min(x0, self._w - w)), max(0, min(y0, self._h - h))

    def _on_press(self, event):
        if event.inaxes is None:
            return
        for rect in self._rects:
            if rect.axes is not event.inaxes:
                continue
            contains, _ = rect.contains(event)
            if contains:
                x0, y0 = rect.get_xy()
                self._press = (x0, y0, event.xdata, event.ydata)
                return

    def _on_release(self, _event):
        self._press = None

    def _on_motion(self, event):
        if self._press is None or event.inaxes is None:
            return
        x0, y0, xpress, ypress = self._press
        r = self._rects[0]
        w, h = r.get_width(), r.get_height()
        new_x, new_y = self._clamp(x0 + event.xdata - xpress, y0 + event.ydata - ypress, w, h)
        for rect in self._rects:
            rect.set_xy((new_x, new_y))
        for ax in self._zoom_axes:
            ax.set_xlim(new_x, new_x + w)
            ax.set_ylim(new_y, new_y + h)
        r.figure.canvas.draw_idle()


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
        self._zoom_drag = None  # keep drag handler alive with the tab

    def set_figure(self, fig: Figure):
        if self._canvas is not None:
            self._lay.removeWidget(self._toolbar)
            self._lay.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()
        self._zoom_drag = None
        self._placeholder.setVisible(False)
        self._canvas = FigureCanvasQTAgg(fig)
        # Wire drag handler now that fig.canvas points to the Qt canvas
        if hasattr(fig, '_zoom_drag_data'):
            rects, zoom_axes, shape = fig._zoom_drag_data
            self._zoom_drag = _DraggableZoomRect(rects, zoom_axes, shape)
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

        self._locs_tab = _FigureTab("Run fitting to see localisations.")
        self._tabs.addTab(self._locs_tab, "Localisations")

        self._stats_tab = _FigureTab("Run fitting to see photon statistics.")
        self._tabs.addTab(self._stats_tab, "Statistics")

    def set_preview_figure(self, fig: Figure):
        self._preview_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._preview_tab)

    def set_localisations_figure(self, fig: Figure):
        self._locs_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._locs_tab)

    def set_stats_figure(self, fig: Figure):
        self._stats_tab.set_figure(fig)
