import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from soche.chords import get_chords_file


def create_app(audio_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    cached_chords = get_chords_file(audio_path) if audio_path else []

    @app.get("/")
    def index():
        return render_template("viewer.html")

    @app.get("/chords")
    def get_chords():
        return jsonify(cached_chords)

    @app.post("/chords")
    def post_chords():
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fp:
                fp.write(request.data)
            return jsonify(get_chords_file(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @app.get("/audio")
    def get_audio():
        if not audio_path or not audio_path.is_file():
            return ("Not found", 404)
        return send_file(audio_path.resolve(), conditional=True)

    return app


def main() -> None:
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    port = int(os.environ.get("PORT", 8000))
    create_app(audio_path).run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
