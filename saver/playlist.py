from __future__ import annotations

from pathlib import Path

from .models import Track
from .vision import VisionConfig, VisionExtractionError, extract_tracks_from_screenshots, resolve_screenshot_paths


class PlaylistSourceError(RuntimeError):
    """Raised when screenshot input cannot be resolved into tracks."""


def load_tracks(
    *,
    screenshots: list[str] | None = None,
    screenshot_dir: str | None = None,
    vision_model: str = "llava:latest",
    ollama_url: str = "http://127.0.0.1:11434",
    vision_timeout: float = 180.0,
) -> tuple[list[Track], dict]:
    try:
        screenshot_paths = resolve_screenshot_paths(
            [Path(path) for path in screenshots or []],
            Path(screenshot_dir) if screenshot_dir else None,
        )
        config = VisionConfig(
            model=vision_model,
            ollama_url=ollama_url,
            timeout=vision_timeout,
        )
        tracks = extract_tracks_from_screenshots(screenshot_paths, config=config)
    except VisionExtractionError as exc:
        raise PlaylistSourceError(str(exc)) from exc

    return tracks, {
        "title": "Screenshot playlist",
        "url": None,
        "source": "screenshots",
        "screenshot_count": len(screenshot_paths),
        "vision_model": vision_model,
    }
