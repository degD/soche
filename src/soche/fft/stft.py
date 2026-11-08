
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from soche.wav.constants import CHORDS_MINIMAL, SAMPLE_RATE
from soche.wav.sine import generate_chord, multi_array_sum
from soche.wav.write_wave import generate_sin_array_normalized


def stft(signal):
    frames = sliding_window_view(signal, 4000)[::500]   # 4000 is window size, 500 is hop size
    frames_windowed = frames * np.hanning(4000)         # TODO: sliding view funcion is slow for prod
    return np.fft.rfft(frames_windowed)


def plot_stft(signal):
    signal_stft = stft(signal)
    plt.plot(signal_stft[10])
    plt.show()


if __name__ == "__main__":
    f = generate_sin_array_normalized(0.5, 100, 2)
    print(stft(f))
    plot_stft(f)
