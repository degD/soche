
import numpy as np
import numpy.typing as npt
from soche.wav.constants import CHORDS_MINIMAL, NOTES_FREQ_MINIMAL


# a frame is a 1xF vector, F is number of frequency bins
# a chroma_frame is a 1x12 vector, 12 is number of notes


NOTE_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def chord_to_template(chord: str) -> npt.NDArray:
    notes_str = CHORDS_MINIMAL[chord]
    t = [0.0] * 12
    for n in notes_str.split(","):
        t[NOTE_TO_INDEX[n]] = 1.0
    return np.array(t)
    # TODO: give higher score to root


def cosine_similarity_frame_and_chord(frame: npt.NDArray, chord) -> float:
    # chord_template = chord_to_template(chord)
    dot_prod = np.dot(frame, chord)
    mag_mul = np.sqrt(np.dot(frame, frame)) * np.sqrt(np.dot(chord, chord))
    score = dot_prod / mag_mul
    return score
    # TODO: could be a better way to calc cosine similarity


if __name__ == "__main__":
    frame = [1, 4, 1, 0, 5]
    c1    = [0, 0, 1, 0, 1]
    c2    = [1, 1, 0, 0, 0]
    c3    = [0, 1, 0, 0, 1]
    print(f"c1: {cosine_similarity_frame_and_chord(np.array(frame), np.array(c1))}")
    print(f"c2: {cosine_similarity_frame_and_chord(np.array(frame), np.array(c2))}")
    print(f"c3: {cosine_similarity_frame_and_chord(np.array(frame), np.array(c3))}")
