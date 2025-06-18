#!/usr/bin/env python
import os
import sys
from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtWidgets import QApplication
from interface.video_window import VideoWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Qt'nin plugin klasörü yolu:
    plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    # Ortam değişkenine kayıt
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    
    player = VideoWindow()
    player.showMaximized()
    sys.exit(app.exec_())
