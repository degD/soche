import json
import tempfile

from flask import Flask, render_template, request

from soche.chords.output import get_chords_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/get_chords", methods=["POST"])
def get_chords():

    # write opened file data as .wav tmp file
    _, path = tempfile.mkstemp(suffix=".wav")
    with open(path, "wb") as fp:
        fp.write(request.data)

    # read wave file and return chords
    chords_and_counts = get_chords_file(path)
    return json.dumps(chords_and_counts)

app.run(debug=True)
