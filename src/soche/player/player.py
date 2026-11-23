import json
import os
import tempfile

from flask import Flask, render_template, request

from soche.chords.output import get_chords_file

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_chords", methods=["POST"])
def get_chords():
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(request.data)
        chords_and_counts = get_chords_file(path)
        return json.dumps(chords_and_counts)
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    app.run(debug=True)
