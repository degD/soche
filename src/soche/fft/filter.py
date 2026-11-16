
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view


def harmony_median_filter(stft_signal: npt.NDArray, filter_size: int) -> npt.NDArray:
    pad_size = filter_size // 2
    padded_stft_signal = np.pad(stft_signal, ((pad_size, pad_size), (0, 0)), mode="edge")
    windows = sliding_window_view(padded_stft_signal, filter_size, axis=0)
    return np.median(windows, axis=2)


if __name__ == "__main__":
    demo_signal = np.array([
        [5, 5, 5],
        [0, 0, 0],
        [1, 2, 3],
        [0, 0, 0],
        [0, 0, 0],
        [1, 2, 3],
    ])

    print(harmony_median_filter(demo_signal, 3))
