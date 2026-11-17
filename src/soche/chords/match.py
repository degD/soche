
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from soche.chords.templates import CHORD_TEMPLATES
from soche.fft.filter import harmony_median_filter
from soche.fft.stft import calc_chroma, stft
from soche.wav import read_wave

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
    return np.array(CHORD_TEMPLATES[chord])


def cosine_similarity_frame_and_chord(frame: npt.NDArray, chord: str) -> float:
    chord_template = chord_to_template(chord)
    dot_prod = np.dot(frame, chord_template)
    mag_mul = np.sqrt(np.dot(frame, frame)) * np.sqrt(np.dot(chord_template, chord_template))
    score = dot_prod / mag_mul
    return score
    # TODO: could be a better way to calc cosine similarity
    # TODO: Maybe multiply with matrix to calc faster?


def get_chords_list() -> list[str]:
    return list(CHORD_TEMPLATES)


def cosine_similarity_chroma(chroma: npt.NDArray) -> list[tuple[int, float]]:
    chord_classes = []
    # TODO: Any better way than chroma.T surely exist!
    for frame in chroma.T:
        scores = []
        for chord in get_chords_list():
            s = cosine_similarity_frame_and_chord(frame, chord)
            scores.append(s)
        max_s = max(scores)
        i = scores.index(max_s)
        chord_classes.append((i, max_s))
    return chord_classes


def plot_chord_matching(chord_matching_list: list[tuple[int, float]]) -> None:
    chord_labels = get_chords_list()
    grid = np.zeros((len(chord_labels), len(chord_matching_list)))
    for frame_idx, (chord_idx, score) in enumerate(chord_matching_list):
        grid[chord_idx, frame_idx] = score

    plt.imshow(
        grid,
        aspect="auto",
    )
    plt.yticks(ticks=np.arange(len(chord_labels)), labels=chord_labels)
    plt.xlabel("Frame")
    plt.ylabel("Chord")
    plt.grid()
    plt.show()
    plt.close()


if __name__ == "__main__":
    window_size = 4096
    hop_size = 512
    ref_freq = 440

    # f = test_signals.spanish()
    f = read_wave.read_wave("audio/rec1.wav")
    stft_f = stft(f, window_size, hop_size)
    filtered_f = harmony_median_filter(stft_f, 15)
    chroma_f = calc_chroma(filtered_f, window_size, ref_freq)
    chord_matching_list = cosine_similarity_chroma(chroma_f)
    # plot_chroma(chroma_f)
    plot_chord_matching(chord_matching_list)
