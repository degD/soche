
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from soche.fft import test_signals
from soche.wav.constants import SAMPLE_RATE

# for a signal of n samples, hightest freq is sample_rate / 2.
# there are 16_000 samples each second. for representing a wave.
# you need at least 2 samples, one for up and one for down.
# therefore 100 sample_rate for 50Hz, 16_000 sample_rate for 8000Hz.


def fft_signal_and_plot(signal: npt.NDArray[np.float32], min_freq: float, max_freq: float) -> None:
    n = len(signal)                             # dividing by n normalizes freq magnitues
    y = np.abs(np.fft.rfft(signal)) / n         # fft results complex numbers and abs returns magnitudes
    x = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)     # each sample is (1 / 16000) seconds after the previous one
    plt.plot(x, y)
    plt.xlim(min_freq, max_freq)
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.show()
    plt.close()


def fft_signal_and_specgram(
    signal: npt.NDArray[np.float32],
    min_freq: float,
    max_freq: float,
    window_size: int,
    hop_size: int
) -> None:
    plt.specgram(
        signal,                           # specgram can run fft on its own
        NFFT=window_size,                 # Fs is sample rate, NFFT is slice size
        Fs=SAMPLE_RATE,                   # large NFFT: higher freq resolution, but poor time resolution
        noverlap=window_size - hop_size,  # noverlap = window_size - hop_size
        window=np.hanning(window_size)    # upright lines are a result of synthetic sound generation
    )
    plt.grid()
    plt.ylim(min_freq, max_freq)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.show()
    plt.close()


if __name__ == "__main__":
    f = test_signals.spanish()
    fft_signal_and_specgram(f, 200, 600, 2048, 1024)
