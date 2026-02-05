"""Launcher script for LC-MS Analysis app."""

import sys
import os
import time
import threading

class LoadingSpinner:
    """Display a spinning animation with timer during loading."""
    def __init__(self):
        self.running = False
        self.start_time = None
        self.thread = None
        self.message = "Loading"
        # Spinning circle animation frames
        self.frames = ["◐", "◓", "◑", "◒"]

    def _update(self):
        frame_idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.frames[frame_idx % len(self.frames)]
            print(f"\r  {spinner} {self.message}  {elapsed:.1f}s", end="  ")
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.15)

    def start(self, message="Loading"):
        self.message = message
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        elapsed = time.time() - self.start_time
        print(f"\r  ✓ {self.message}  {elapsed:.1f}s   ")

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

    print()
    print("  LC-MS Analysis")
    print("  ─────────────────────────")

    spinner = LoadingSpinner()

    # Pre-load heavy libraries with spinner + timer
    spinner.start("Loading")
    import numpy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy
    import streamlit
    spinner.stop()

    print()
    print(f"  URL: http://localhost:8501")
    print("  ─────────────────────────")
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
