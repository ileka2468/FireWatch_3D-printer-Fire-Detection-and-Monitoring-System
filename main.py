"""
3D Print Fire Detection System
==================================

A desktop application that monitors multiple network camera streams
and uses local AI vision (Gemma 4 12B via Ollama) to detect fire and
smoke near 3D printers.

Requirements:
    - Python 3.10+
    - Ollama running locally with gemma4:12b pulled
    - pip install customtkinter opencv-python Pillow ollama

Usage:
    python main.py
"""

import logging
import sys
import os

# Resolve the application directory (works for both script and PyInstaller exe)
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Set working directory to app location for relative paths
os.chdir(APP_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(APP_DIR, "firedetection.log"),
            mode="a", encoding="utf-8"
        )
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("[FIREWATCH] 3D Print Fire Detection System starting...")
    logger.info(f"App directory: {APP_DIR}")
    logger.info("=" * 50)

    try:
        from gui.app import FireDetectionApp
        app = FireDetectionApp()
        app.mainloop()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application exited.")


if __name__ == "__main__":
    main()
