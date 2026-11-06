
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt


def info_wave(wave_path: str) -> None:
    with wave.open(wave_path, "rb") as fp:
        _channels = fp.getnchannels()
        _sample_width = fp.getsampwidth()    # sample size, number of bytes required to represent a single sample
        _framerate = fp.getframerate()       # number of samples / seconds
        _num_frames = fp.getnframes()        # total number of samples in audio
        print(fp.getparams())


def normalize_int16_array(array: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    max_val = np.iinfo(np.int16).max
    min_val = np.iinfo(np.int16).min
    array_int32 = np.astype(array, np.int32)
    array_float64 = 2 * (array_int32 - min_val) / (max_val - min_val) - 1
    array_float32 = np.astype(array_float64, np.float32)
    return array_float32


def read_wave(wave_path: str | Path) -> npt.NDArray[np.float32]:
    with wave.open(str(wave_path), "rb") as fp:
        num_of_frames = fp.getnframes();
        buffer = fp.readframes(num_of_frames)   # TODO: parse loop in fixed sizes
        data = np.frombuffer(buffer, "<i2")     # wav is little endian
        return normalize_int16_array(data)      # float32 is smallest possible size


if __name__ == "__main__":
    # info_wave("audio/audio.wav")
    print(read_wave("audio/audio.wav"))
