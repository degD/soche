
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from soche.fft.test_signals import spanish


def stft(signal: npt.NDArray[np.float32], window_size: int, hop_size: int) -> npt.NDArray:
    stft_frames = sliding_window_view(signal, window_size)[::hop_size]
    frames_windowed = stft_frames * np.hanning(window_size)     # TODO: sliding view funcion is slow for prod
    return np.abs(np.fft.rfft(frames_windowed))            # results in (num_frames, freq_bins) matrix
                                                                # square to get power specgram


def calc_chroma(stft_matrix: npt.NDArray, window_size: int, ref_freq: int = 440) -> npt.NDArray:
    freqs = np.fft.rfftfreq(window_size, 1 / 44_100)   # get frequency values for the range
    filterbank = np.zeros((12, len(freqs)))                 # filtering, there are 12 notes
    for i in range(1, len(freqs)):                              # convert frequency values to midi note numbers, and to pitch classes (notes)
        midi = round(69 + 12 * np.log2(freqs[i] / ref_freq))    # puts all frequencies to matching bins
        pitch_class = midi % 12
        filterbank[pitch_class, i] = 1.0                # filterbank is (12, freq_bins) and stft_matrix is (num_frames, freq_bins)
    chroma = filterbank @ stft_matrix.T  # get energy levels for each note for each frame
    chroma /= chroma.max(axis=0, keepdims=True) + 1e-6  # normalize chroma values to 0..1 with max energy levels on each frame
    return chroma

# NOTE: Considering 440 is A, round(69 + 12 * log2(440 / 440)) -> round(69 + 12 * log2(1)) -> round(69) -> 69
# midi = 69 -> pitch_class = 9
# 0: C, 1: C#, 2: D ... 9: A, 10: A#, 11: B


def plot_chroma(chroma: npt.NDArray) -> None:
    pitch_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    plt.imshow(
        chroma,
        aspect="auto",
    )
    plt.yticks(ticks=np.arange(12), labels=pitch_labels)
    plt.xlabel("Frame")
    plt.ylabel("Pitch Class")
    plt.grid()
    plt.show()
    plt.close()


if __name__ == "__main__":
    window_size = 4096
    hop_size = 512
    ref_freq = 400
    # f = read_wave("audio/guitar_prog_web.wav")
    f = spanish()
    plot_chroma(calc_chroma(stft(f, window_size, hop_size), window_size, ref_freq))
