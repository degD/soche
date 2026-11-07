
import numpy as np
import numpy.typing as npt

from soche.wav.constants import CHORDS_MINIMAL, NOTES_FREQ_MINIMAL, SAMPLE_RATE
from soche.wav.write_wave import generate_sin_array_normalized, write_wave


def generate_sin_array_normalized_with_shift(amp: float, freq: float, duration_sec: float, shift_sec: float) -> npt.NDArray[np.float32]:
    shift = np.zeros(int(SAMPLE_RATE * shift_sec), np.float32)
    array = generate_sin_array_normalized(amp, freq, duration_sec)
    return np.concat((shift, array))


def array_sum(a1: npt.NDArray[np.float32], a2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    max_len = max(len(a1), len(a2))
    a1 = np.pad(a1, (0, max_len - len(a1)))
    a2 = np.pad(a2, (0, max_len - len(a2)))
    return a1 + a2


def multi_array_sum(arrays: tuple[npt.NDArray[np.float32], ...]) -> npt.NDArray[np.float32]:
    s = np.zeros(0, np.float32)
    for i in range(len(arrays)):
        s = array_sum(s, arrays[i])
    return s


def generate_chord(amp: float, notes_str: str, duration_sec: float, shift_sec: float) -> npt.NDArray[np.float32]:
    chords = []
    for n in notes_str.split(","):
        s = generate_sin_array_normalized_with_shift(amp, NOTES_FREQ_MINIMAL[n], duration_sec, shift_sec)
        chords.append(s)
    return multi_array_sum(tuple(chords))


if __name__ == "__main__":
    am = generate_chord(0.1, CHORDS_MINIMAL["Am"], 1, 0)
    g  = generate_chord(0.1, CHORDS_MINIMAL["G"], 1, 1)
    f  = generate_chord(0.1, CHORDS_MINIMAL["F"], 1, 2)
    e  = generate_chord(0.1, CHORDS_MINIMAL["E"], 1, 3)

    f = multi_array_sum((e, am, f, g))
    write_wave("audio/progression.wav", f)
