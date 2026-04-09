from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .models import MapEntry, MapVersion

BASE_URL = "https://api.beatsaver.com"
USER_AGENT = "beatmapcollecter/0.1"


class BeatSaverClient:
    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def search_text(self, query: str, page: int = 0) -> list[MapEntry]:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"{BASE_URL}/search/text/{page}?q={encoded_query}"
        payload = self._get_json(url)
        docs = payload.get("docs") or payload.get("maps") or payload.get("results") or []
        return [self._parse_map_entry(doc) for doc in docs if isinstance(doc, dict)]

    def download_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _get_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_map_entry(self, payload: dict) -> MapEntry:
        metadata = payload.get("metadata") or {}
        versions: list[MapVersion] = []
        for version in payload.get("versions", []):
            if not isinstance(version, dict):
                continue
            download_url = version.get("downloadURL") or version.get("downloadUrl")
            hash_value = version.get("hash") or ""
            if not download_url:
                continue
            versions.append(
                MapVersion(
                    hash_value=hash_value,
                    download_url=download_url,
                    diffs=[
                        str(diff.get("difficulty"))
                        for diff in version.get("diffs", [])
                        if isinstance(diff, dict) and diff.get("difficulty")
                    ],
                )
            )

        return MapEntry(
            map_id=str(payload.get("id") or ""),
            song_name=str(metadata.get("songName") or payload.get("name") or ""),
            song_sub_name=str(metadata.get("songSubName") or ""),
            song_author_name=str(metadata.get("songAuthorName") or ""),
            level_author_name=str(metadata.get("levelAuthorName") or ""),
            versions=versions,
            raw=payload,
        )
