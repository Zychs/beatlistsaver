from __future__ import annotations

import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .bundler import build_bundle
from .playlist import load_tracks


def serve(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send_html(_render_page())

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length).decode("utf-8")
            form = {key: values[0] for key, values in parse_qs(payload).items()}
            try:
                include_all_versions = form.get("include_all_versions") == "on"
                tracks, playlist_info = load_tracks(
                    screenshots=_split_screenshot_paths(form.get("screenshot_paths") or ""),
                    screenshot_dir=form.get("screenshot_dir") or None,
                    vision_model=form.get("vision_model") or "llava:latest",
                    ollama_url=form.get("ollama_url") or "http://127.0.0.1:11434",
                    vision_timeout=float(form.get("vision_timeout") or "180"),
                )
                output_name = (
                    form.get("output_name") or ""
                ).strip() or "beat-saber-playlist-bundle.zip"
                output_path = Path.home() / "Desktop" / output_name
                result = build_bundle(
                    tracks,
                    output_path=output_path,
                    playlist_info=playlist_info,
                    min_score=float(form.get("min_score") or "0.74"),
                    max_pages=max(1, int(form.get("max_pages") or "2")),
                    include_all_versions=include_all_versions,
                )
                body = _render_page(
                    result_html=_render_success(result.output_path, result.manifest),
                    values=form,
                )
                self._send_html(body)
            except Exception as exc:  # noqa: BLE001
                body = _render_page(
                    result_html=f"<section class='result error'><h2>Run failed</h2><p>{html.escape(str(exc))}</p></section>",
                    values=form,
                )
                self._send_html(body, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _render_success(output_path: Path, manifest: dict) -> str:
    return f"""
    <section class="result success">
      <h2>Bundle ready</h2>
      <p><strong>Saved to Desktop as:</strong> {html.escape(output_path.name)}</p>
      <p><strong>Tracks scanned:</strong> {manifest["track_count"]} |
      <strong>Matched:</strong> {manifest["matched_track_count"]} |
      <strong>Variants:</strong> {manifest["downloaded_variant_count"]}</p>
    </section>
    """


def _render_page(*, result_html: str = "", values: dict | None = None) -> str:
    values = values or {}

    def fill(name: str, default: str = "") -> str:
        return html.escape(values.get(name, default))

    if values.get("submitted") == "1":
        checked = "checked" if values.get("include_all_versions") == "on" else ""
    else:
        checked = "checked"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Beat Playlist Zipper</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe7;
      --panel: rgba(255, 251, 246, 0.92);
      --ink: #241c17;
      --muted: #64584f;
      --line: rgba(85, 62, 47, 0.14);
      --accent: #b94f2f;
      --accent-2: #224c59;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(185,79,47,.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(34,76,89,.14), transparent 28%),
        linear-gradient(180deg, #fbf6ef, var(--bg));
      color: var(--ink);
      font-family: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
    }}
    main {{
      width: min(900px, calc(100vw - 24px));
      margin: 20px auto 40px;
      display: grid;
      gap: 18px;
    }}
    .panel, .result {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 26px;
      padding: 22px;
      box-shadow: 0 24px 60px rgba(55,39,28,.12);
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.3rem, 7vw, 4rem);
      line-height: .95;
      letter-spacing: -.04em;
      max-width: 10ch;
    }}
    p {{ line-height: 1.6; color: var(--muted); }}
    form {{ display: grid; gap: 14px; }}
    label {{ display: grid; gap: 8px; font-weight: 700; color: var(--accent-2); }}
    input, textarea {{
      width: 100%;
      font: inherit;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.84);
      padding: 14px;
      color: var(--ink);
    }}
    textarea {{ min-height: 220px; resize: vertical; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .check {{ display: flex; align-items: center; gap: 10px; font-weight: 600; color: var(--ink); }}
    .check input {{ width: auto; }}
    button {{
      width: fit-content;
      border: 0;
      border-radius: 999px;
      padding: 13px 20px;
      font: inherit;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #d66f41);
      color: white;
      cursor: pointer;
    }}
    .success {{ border-color: rgba(36,90,56,.2); }}
    .error {{ border-color: rgba(148,47,47,.24); }}
    code {{
      background: rgba(34,76,89,.08);
      padding: 2px 6px;
      border-radius: 999px;
    }}
    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Beat Playlist Zipper</h1>
      <p>Point this at high-resolution playlist screenshots. It reads visible tracks locally, searches BeatSaver, pulls every matched map variant, and writes one desktop zip.</p>
      <p>Start Ollama first and use a vision model such as <code>llava:latest</code>.</p>
    </section>
    {result_html}
    <section class="panel">
      <form method="post">
        <input name="submitted" type="hidden" value="1">
        <label>Screenshot paths
          <textarea name="screenshot_paths" placeholder="C:\\path\\playlist-page-1.png&#10;C:\\path\\playlist-page-2.png">{fill("screenshot_paths")}</textarea>
        </label>
        <label>Screenshot directory
          <input name="screenshot_dir" type="text" value="{fill("screenshot_dir")}" placeholder="C:\\path\\playlist-screenshots">
        </label>
        <div class="grid">
          <label>Output zip name
            <input name="output_name" type="text" value="{fill("output_name", "beat-saber-playlist-bundle.zip")}">
          </label>
          <label>Vision model
            <input name="vision_model" type="text" value="{fill("vision_model", "llava:latest")}">
          </label>
          <label>Ollama URL
            <input name="ollama_url" type="text" value="{fill("ollama_url", "http://127.0.0.1:11434")}">
          </label>
          <label>Vision timeout seconds
            <input name="vision_timeout" type="number" min="10" max="600" step="1" value="{fill("vision_timeout", "180")}">
          </label>
          <label>Minimum match score
            <input name="min_score" type="number" min="0" max="1" step="0.01" value="{fill("min_score", "0.74")}">
          </label>
          <label>Search pages per query
            <input name="max_pages" type="number" min="1" max="5" step="1" value="{fill("max_pages", "2")}">
          </label>
        </div>
        <label class="check"><input name="include_all_versions" type="checkbox" {checked}>Include every version for each matched BeatSaver map</label>
        <button type="submit">Build desktop zip</button>
      </form>
    </section>
  </main>
</body>
</html>"""


def _split_screenshot_paths(value: str) -> list[str]:
    paths: list[str] = []
    for line in value.splitlines():
        line = line.strip()
        if line:
            paths.append(line)
    return paths


if __name__ == "__main__":
    serve()
