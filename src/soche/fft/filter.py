
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from soche.fft.stft import calc_chroma, plot_chroma, stft
from soche.wav.read_wave import read_wave


def harmony_median_filter(stft_signal: npt.NDArray, filter_size: int) -> npt.NDArray:
    if filter_size < 1 or filter_size % 2 == 0:
        raise ValueError("filter_size must be a positive odd integer!")
    pad_size = filter_size // 2
    padded_stft_signal = np.pad(stft_signal, ((pad_size, pad_size), (0, 0)), mode="edge")
    windows = sliding_window_view(padded_stft_signal, filter_size, axis=0)
    return np.median(windows, axis=2)
    # TODO: Explain and document


if __name__ == "__main__":
    f = read_wave("audio/rec1.wav")
    f = stft(f, 8192, 512)
    f_h = harmony_median_filter(f, 51)
    c_h = calc_chroma(f_h, 8192)
    plot_chroma(c_h)
