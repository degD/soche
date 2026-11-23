import json
import mimetypes
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from soche.chords import get_chords_file

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>soche viewer</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  .now { display: flex; align-items: baseline; gap: 1rem; margin: 1rem 0; }
  h1 { font-size: 4rem; margin: 0; min-width: 7rem; }
  .adj { color: #666; font-size: 1.25rem; }
  .controls { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }
  audio { width: 100%; margin: .5rem 0 1rem; }
  #timeline { display: flex; flex-wrap: wrap; gap: 4px; max-height: 50vh; overflow-y: auto; }
  .seg { border: 1px solid #ccc; background: #f8f8f8; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: .85rem; }
  .seg.active { background: #111; color: #fff; border-color: #111; }
</style>
</head>
<body>
  <div class="controls">
    <input id="file" type="file" accept="audio/*">
    <button onclick="setRate(0.5)">0.5x</button>
    <button onclick="setRate(0.75)">0.75x</button>
    <button onclick="setRate(1)">1x</button>
  </div>
  <audio id="audio" controls></audio>
  <div class="now"><h1 id="chord">N</h1><span class="adj" id="next"></span></div>
  <div id="timeline"></div>
<script>
const audio = document.getElementById("audio");
const chordEl = document.getElementById("chord");
const nextEl = document.getElementById("next");
const timeline = document.getElementById("timeline");
let segs = [], btns = [], active = -1;

function setRate(r) { audio.playbackRate = r; }

function loadSegments(data) {
  segs = data; active = -1; timeline.innerHTML = "";
  btns = segs.map(([s, e, c]) => {
    const b = document.createElement("button");
    b.className = "seg";
    b.textContent = `${s.toFixed(1)}s ${c}`;
    b.onclick = () => { audio.currentTime = s; audio.play(); };
    timeline.appendChild(b);
    return b;
  });
}

function tick() {
  const t = audio.currentTime;
  let idx = segs.findIndex(([s, e]) => t >= s && t < e);
  if (idx !== active) {
    if (active >= 0 && btns[active]) btns[active].classList.remove("active");
    active = idx;
    if (idx >= 0) {
      btns[idx].classList.add("active");
      btns[idx].scrollIntoView({ block: "nearest" });
      chordEl.textContent = segs[idx][2];
      nextEl.textContent = segs[idx + 1] ? `→ ${segs[idx + 1][2]}` : "";
    } else {
      chordEl.textContent = "N";
      nextEl.textContent = "";
    }
  }
  requestAnimationFrame(tick);
}

document.getElementById("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  audio.src = URL.createObjectURL(f);
  chordEl.textContent = "...";
  const res = await fetch("/chords", { method: "POST", body: f });
  loadSegments(await res.json());
};

fetch("/chords").then(r => r.json()).then(data => {
  if (data.length) { audio.src = "/audio"; loadSegments(data); }
});
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def make_handler(audio_path: Path | None) -> type[BaseHTTPRequestHandler]:
    cached_chords = get_chords_file(audio_path) if audio_path else []

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/chords":
                self._send(200, json.dumps(cached_chords).encode("utf-8"), "application/json")
            elif self.path == "/audio" and audio_path and audio_path.is_file():
                data = audio_path.read_bytes()
                ctype = mimetypes.guess_type(audio_path)[0] or "audio/wav"
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start_s, _, end_s = rng[6:].partition("-")
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else len(data) - 1
                    chunk = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            else:
                self._send(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/chords":
                self._send(404, b"Not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = self.rfile.read(length)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, "wb") as fp:
                    fp.write(payload)
                chords = get_chords_file(tmp)
                self._send(200, json.dumps(chords).encode("utf-8"), "application/json")
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    return Handler


def main() -> None:
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("127.0.0.1", port), make_handler(audio_path))
    print(f"Serving on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
