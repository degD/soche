
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from soche.wav.constants import CHORDS_MINIMAL, SAMPLE_RATE
from soche.wav.sine import generate_chord, multi_array_sum
from soche.wav.write_wave import generate_sin_array_normalized


def stft(signal: npt.NDArray[np.float32], window_size: int, hop_size: int):
    stft_frames = sliding_window_view(signal, window_size)[::hop_size]   # TODO: sliding view funcion is slow for prod
    frames_windowed = stft_frames # * np.hanning(window_size)
    return np.fft.rfft(frames_windowed)


def plot_stft(stft_frames):
    fig, ax = plt.subplots()
    ax.specgram()


if __name__ == "__main__":
    f = generate_sin_array_normalized(0.5, 100, 2)
    print(stft(f))
    plot_stft(f)
