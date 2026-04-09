from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Track

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_PROMPT = """Read this playlist screenshot and extract only visible song rows.

Return strict JSON using this shape:
{"tracks":[{"title":"Song title","artists":["Artist name"],"confidence":0.0}]}

Rules:
- Use the actual song title, not album names, durations, menu labels, or app chrome.
- Split multiple artists into separate array values.
- Remove badges like Explicit, official video markers, and track numbers.
- If a row is partially unreadable, include it only when the title is clear.
- Do not invent missing rows or infer songs that are not visible.
- Return JSON only.
"""


class VisionExtractionError(RuntimeError):
    """Raised when local vision extraction fails or returns unusable data."""


@dataclass(slots=True)
class VisionConfig:
    model: str = "llava:latest"
    ollama_url: str = "http://127.0.0.1:11434"
    timeout: float = 180.0
    prompt: str = DEFAULT_PROMPT


class OllamaVisionClient:
    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    def extract_tracks(self, image_path: Path, *, source: str | None = None) -> list[Track]:
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        request_body = {
            "model": self.config.model,
            "prompt": self.config.prompt,
            "images": [encoded_image],
            "format": "json",
            "stream": False,
        }
        request = urllib.request.Request(
            _join_url(self.config.ollama_url, "/api/generate"),
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise VisionExtractionError(
                "Ollama did not answer. Start Ollama, pull a vision model, or set --ollama-url."
            ) from exc
        except json.JSONDecodeError as exc:
            raise VisionExtractionError("Ollama returned invalid JSON from /api/generate.") from exc

        response_text = str(payload.get("response") or "")
        if not response_text.strip():
            raise VisionExtractionError("Ollama returned an empty response for a screenshot.")
        return parse_vision_response(response_text, source=source or "screenshot")


def extract_tracks_from_screenshots(paths: list[Path], *, config: VisionConfig | None = None) -> list[Track]:
    if not paths:
        raise VisionExtractionError("Provide at least one screenshot path.")

    client = OllamaVisionClient(config)
    tracks: list[Track] = []
    for index, path in enumerate(paths, start=1):
        tracks.extend(client.extract_tracks(path, source=f"screenshot:{index}"))

    tracks = _dedupe_tracks(tracks)
    if not tracks:
        raise VisionExtractionError("No tracks were extracted from the screenshot input.")
    return tracks


def resolve_screenshot_paths(paths: list[Path], directory: Path | None = None) -> list[Path]:
    resolved: list[Path] = []

    for path in paths:
        resolved.append(_validate_image_path(path))

    if directory:
        if not directory.exists():
            raise VisionExtractionError("Screenshot directory does not exist.")
        if not directory.is_dir():
            raise VisionExtractionError("Screenshot directory is not a directory.")
        for path in sorted(directory.iterdir()):
            if path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS:
                resolved.append(_validate_image_path(path))

    resolved = _dedupe_paths(resolved)
    if not resolved:
        raise VisionExtractionError("Provide --screenshot or --screenshot-dir with PNG, JPG, WEBP, or BMP files.")
    return resolved


def parse_vision_response(response_text: str, *, source: str = "screenshot") -> list[Track]:
    try:
        payload = json.loads(_strip_json_markdown(response_text))
    except json.JSONDecodeError as exc:
        raise VisionExtractionError("The local vision model did not return valid track JSON.") from exc
    return tracks_from_vision_payload(payload, source=source)


def tracks_from_vision_payload(payload: Any, *, source: str = "screenshot") -> list[Track]:
    items = payload.get("tracks") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise VisionExtractionError("Vision JSON must contain a tracks array.")

    tracks: list[Track] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title") or item.get("song") or item.get("name"))
        if not title:
            continue

        artists = _coerce_artists(item)
        raw = {
            "vision_index": index,
            "confidence": _coerce_confidence(item.get("confidence")),
            "source": source,
        }
        tracks.append(Track(title=title, artists=artists, source=source, raw=raw))

    tracks = _dedupe_tracks(tracks)
    if not tracks:
        raise VisionExtractionError("Vision JSON did not contain any usable tracks.")
    return tracks


def _validate_image_path(path: Path) -> Path:
    if not path.exists():
        raise VisionExtractionError("Screenshot does not exist.")
    if not path.is_file():
        raise VisionExtractionError("Screenshot path is not a file.")
    if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise VisionExtractionError("Unsupported screenshot file type.")
    return path.resolve()


def _coerce_artists(item: dict) -> list[str]:
    raw_artists = item.get("artists")
    if raw_artists is None:
        raw_artists = item.get("artist") or item.get("creator") or item.get("channel")

    artists: list[str] = []
    if isinstance(raw_artists, list):
        for artist in raw_artists:
            artists.extend(_split_artists(str(artist)))
    elif raw_artists:
        artists.extend(_split_artists(str(raw_artists)))

    return _dedupe_text([_clean_text(artist) for artist in artists if _clean_text(artist)])


def _split_artists(value: str) -> list[str]:
    parts = [value.strip()]
    for separator in (",", "&", " feat. ", " feat ", " ft. ", " ft ", " x "):
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(segment.strip() for segment in part.split(separator))
        parts = next_parts
    return [part for part in parts if part]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\bexplicit\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\t\r\n")


def _coerce_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _dedupe_tracks(tracks: list[Track]) -> list[Track]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    output: list[Track] = []
    for track in tracks:
        key = (
            track.title.casefold(),
            tuple(artist.casefold() for artist in track.artists),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(track)
    return output


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _strip_json_markdown(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
