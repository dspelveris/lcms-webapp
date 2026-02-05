"""Launcher script for LC-MS Analysis app."""

import sys
import os
import time
import threading

class LoadingSpinner:
    """Display a colorful spinning animation with timer during loading."""

    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self):
        self.running = False
        self.start_time = None
        self.thread = None
        self.message = "Loading"
        # Colorful spinning animation
        self.frames = [
            f"{self.CYAN}⣾{self.RESET}",
            f"{self.CYAN}⣽{self.RESET}",
            f"{self.BLUE}⣻{self.RESET}",
            f"{self.BLUE}⢿{self.RESET}",
            f"{self.MAGENTA}⡿{self.RESET}",
            f"{self.MAGENTA}⣟{self.RESET}",
            f"{self.CYAN}⣯{self.RESET}",
            f"{self.CYAN}⣷{self.RESET}",
        ]

    def _update(self):
        frame_idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.frames[frame_idx % len(self.frames)]
            timer = f"{self.YELLOW}{elapsed:.1f}s{self.RESET}"
            print(f"\r  {spinner} {self.message}  {timer}", end="    ")
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.1)

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
        check = f"{self.GREEN}✓{self.RESET}"
        timer = f"{self.GREEN}{elapsed:.1f}s{self.RESET}"
        print(f"\r  {check} {self.message}  {timer}    ")

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

    # Colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print()
    print(f"  {BOLD}{CYAN}LC-MS Analysis{RESET}")
    print(f"  {DIM}v1.0{RESET}")
    print()

    # Wait for server to be ready, then open browser
    import webbrowser
    import urllib.request

    def wait_and_open_browser():
        """Wait for server to respond, then open browser."""
        url = "http://localhost:8501"
        max_wait = 30  # Maximum seconds to wait
        start = time.time()

        while time.time() - start < max_wait:
            try:
                urllib.request.urlopen(url, timeout=1)
                # Server is ready - open browser
                webbrowser.open(url)
                return
            except Exception:
                time.sleep(0.5)

    # Start spinner and browser opener
    spinner = LoadingSpinner()
    spinner.start("Starting server")

    def open_when_ready():
        wait_and_open_browser()
        spinner.stop()

    threading.Thread(target=open_when_ready, daemon=True).start()

    print(f"  {CYAN}http://localhost:8501{RESET}")
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
