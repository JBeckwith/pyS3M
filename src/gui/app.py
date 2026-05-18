import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

from gui.main_window import MainWindow


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("pyS3M")
    app.setOrganizationName("LeeGroup")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
