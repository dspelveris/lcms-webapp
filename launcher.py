"""Launcher script for LC-MS Analysis app."""

import sys
import os

# Required for PyInstaller multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Set up paths for frozen executable
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app', 'main.py')
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app', 'main.py')

    # Change to app directory
    os.chdir(os.path.dirname(app_path))

    print("=" * 50)
    print("LC-MS Analysis")
    print("=" * 50)
    print(f"Starting server...")
    print(f"Open in browser: http://localhost:8501")
    print("=" * 50)
    print()

    # Run streamlit directly (not via subprocess)
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false"
    ]

    from streamlit.web import cli as stcli
    stcli.main()
