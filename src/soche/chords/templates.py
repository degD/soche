
import numpy as np
import numpy.typing as npt

from soche.fft import test_signals
from soche.fft.stft import calc_chroma, stft
from soche.wav.constants import CHORDS_MINIMAL

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


def cosine_similarity_frame_and_chord(frame: npt.NDArray, chord: str) -> float:
    chord_template = chord_to_template(chord)
    dot_prod = np.dot(frame, chord_template)
    mag_mul = np.sqrt(np.dot(frame, frame)) * np.sqrt(np.dot(chord_template, chord_template))
    score = dot_prod / mag_mul
    return score
    # TODO: could be a better way to calc cosine similarity
    # TODO: Maybe multiply with matrix to calc faster?


def get_chords_list() -> list[str]:
    return list(CHORDS_MINIMAL)
    # TODO: RN 24 chords so 24 classes, no empty class


def cosine_similarity_chroma(chroma: npt.NDArray) -> list[int]:
    chord_classes = []
    for frame in chroma.T:
        scores = []
        for chord in get_chords_list():
            s = cosine_similarity_frame_and_chord(frame, chord)
            scores.append(s)
        i = scores.index(max(scores))
        chord_classes.append(i)
    return chord_classes


if __name__ == "__main__":
    window_size = 4096
    hop_size = 512
    ref_freq = 440

    f = test_signals.spanish()
    stft_f = stft(f, window_size, hop_size)
    chroma_f = calc_chroma(stft_f, window_size, ref_freq)
    print(cosine_similarity_chroma(chroma_f))
