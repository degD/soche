
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from soche.wav.constants import SAMPLE_RATE
from soche.wav.write_wave import generate_sin_array_normalized

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


if __name__ == "__main__":
    f = generate_sin_array_normalized(0.5, 220, 2)
    fft_signal_and_plot(f, 200, 400)
