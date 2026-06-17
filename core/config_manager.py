"""
Config Manager — handles reading and writing configuration in INI format.

Resolves the config path relative to the executable location so it works
correctly when packaged with PyInstaller.

INI format:
    [general]
    detection_interval = 5
    alert_cooldown = 30
    model = gemma4:12b
    save_alert_frames = true

    [camera:Printer 1]
    url = rtsp://192.168.1.100:554/stream

    [camera:My Webcam]
    url = 0
"""

import configparser
import sys
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_app_dir() -> Path:
    """
    Get the application directory — works both in development
    and when packaged as a PyInstaller executable.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle — exe is in dist/firedetection/
        return Path(sys.executable).parent
    else:
        # Running as a script
        return Path(os.path.dirname(os.path.abspath(__file__))).parent


def get_resource_path(relative_path: str) -> Path:
    """Get the absolute path to a resource, works for dev and PyInstaller."""
    try:
        # PyInstaller stores bundled data here
        base_path = sys._MEIPASS
    except Exception:
        base_path = get_app_dir()
    return Path(base_path) / relative_path


CONFIG_FILENAME = "firedetection.ini"


def get_config_path() -> Path:
    """Get the full path to the config file."""
    return get_app_dir() / CONFIG_FILENAME


def load_config() -> dict:
    """
    Load configuration from the INI file.
    Returns a dict with 'streams' list and settings.
    """
    config_path = get_config_path()
    result = {
        "streams": [],
        "detection_interval": 5,
        "alert_cooldown": 30,
        "model": "gemma4:12b",
        "save_alert_frames": True,
        "enable_dashboard": False,
        "dashboard_port": 5050
    }

    if not config_path.exists():
        logger.info(f"No config file found at {config_path}, using defaults")
        return result

    try:
        parser = configparser.ConfigParser()
        parser.read(str(config_path), encoding="utf-8")

        # Read general settings
        if parser.has_section("general"):
            general = parser["general"]
            result["detection_interval"] = general.getint("detection_interval", 5)
            result["alert_cooldown"] = general.getint("alert_cooldown", 30)
            result["model"] = general.get("model", "gemma4:12b")
            result["save_alert_frames"] = general.getboolean("save_alert_frames", True)
            result["enable_dashboard"] = general.getboolean("enable_dashboard", False)
            result["dashboard_port"] = general.getint("dashboard_port", 5050)

        # Read camera streams — sections prefixed with "camera:"
        for section in parser.sections():
            if section.startswith("camera:"):
                camera_name = section[len("camera:"):]
                url = parser.get(section, "url", fallback="")
                zoom = parser.getfloat(section, "zoom", fallback=1.0)
                pan_x = parser.getfloat(section, "pan_x", fallback=0.0)
                pan_y = parser.getfloat(section, "pan_y", fallback=0.0)
                brightness = parser.getint(section, "brightness", fallback=0)
                contrast = parser.getfloat(section, "contrast", fallback=1.0)
                
                if camera_name and url:
                    result["streams"].append({
                        "name": camera_name, 
                        "url": url,
                        "zoom": zoom,
                        "pan_x": pan_x,
                        "pan_y": pan_y,
                        "brightness": brightness,
                        "contrast": contrast
                    })

        logger.info(f"Loaded config from {config_path}: "
                    f"{len(result['streams'])} camera(s), "
                    f"interval={result['detection_interval']}s")

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")

    return result


def save_config(streams: list[dict], detection_interval: int,
                alert_cooldown: int, model: str = "gemma4:12b",
                save_alert_frames: bool = True, enable_dashboard: bool = False,
                dashboard_port: int = 5050):
    """
    Save configuration to the INI file.

    Args:
        streams: List of dicts with 'name' and 'url' keys
        detection_interval: Detection check interval in seconds
        alert_cooldown: Alert cooldown in seconds
        model: Ollama model name
        save_alert_frames: Whether to save frames that trigger alerts
        enable_dashboard: Whether the web dashboard is enabled
        dashboard_port: Web dashboard port
    """
    config_path = get_config_path()

    try:
        parser = configparser.ConfigParser()

        # General settings
        parser["general"] = {
            "detection_interval": str(detection_interval),
            "alert_cooldown": str(alert_cooldown),
            "model": model,
            "save_alert_frames": str(save_alert_frames).lower(),
            "enable_dashboard": str(enable_dashboard).lower(),
            "dashboard_port": str(dashboard_port)
        }

        # Camera streams
        for stream in streams:
            section = f"camera:{stream['name']}"
            parser[section] = {
                "url": stream["url"],
                "zoom": str(stream.get("zoom", 1.0)),
                "pan_x": str(stream.get("pan_x", 0.0)),
                "pan_y": str(stream.get("pan_y", 0.0)),
                "brightness": str(stream.get("brightness", 0)),
                "contrast": str(stream.get("contrast", 1.0))
            }

        with open(config_path, "w", encoding="utf-8") as f:
            f.write("; FireWatch - 3D Print Fire Detection System\n")
            f.write("; Configuration file - edit cameras and settings below\n")
            f.write("; To add a camera manually, add a [camera:Name] section with url = ...\n\n")
            parser.write(f)

        logger.info(f"Config saved to {config_path}: {len(streams)} camera(s)")

    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
