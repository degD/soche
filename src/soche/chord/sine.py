
import numpy as np
import numpy.typing as npt

from soche.chord.constants import NOTES_FREQ_MINIMAL
from soche.wav.constants import SAMPLE_RATE
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


if __name__ == "__main__":
    c = generate_sin_array_normalized_with_shift(0.1, NOTES_FREQ_MINIMAL["C"], 1, 0)
    e = generate_sin_array_normalized_with_shift(0.2, NOTES_FREQ_MINIMAL["E"], 1, 1)
    g = generate_sin_array_normalized_with_shift(0.3, NOTES_FREQ_MINIMAL["G"], 1, 2)

    chord_c = generate_sin_array_normalized_with_shift(3, NOTES_FREQ_MINIMAL["C"], 3, 3)
    chord_e = generate_sin_array_normalized_with_shift(3, NOTES_FREQ_MINIMAL["E"], 3, 3)
    chord_g = generate_sin_array_normalized_with_shift(3, NOTES_FREQ_MINIMAL["G"], 3, 3)

    chord = multi_array_sum((chord_c, chord_e, chord_g))
    f = multi_array_sum((c, e, g, chord))
    write_wave("audio/c_chord.wav", f)
