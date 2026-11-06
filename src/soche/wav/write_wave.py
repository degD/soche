
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from soche.wav.constants import SAMPLE_RATE


def write_wave_int16(wave_path: str, array: npt.NDArray[np.int16]) -> None:
    with wave.open(wave_path, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(SAMPLE_RATE)
        fp.writeframes(array.astype("<i2").tobytes())  # little endian


def de_normalize_float_array(array_normalized: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
    max_val = np.float32(np.iinfo(np.int16).max)
    clipped = np.clip(array_normalized, -1.0, 1.0)     # avoid int16 overflow
    scaled = np.round(clipped * max_val)               # round to avoid truncation
    return scaled.astype(np.int16)


def write_wave(wave_path: str | Path, array_normalized: npt.NDArray[np.float32]) -> None:
    array_int16 = de_normalize_float_array(array_normalized)
    write_wave_int16(str(wave_path), array_int16)


def generate_sin_array_normalized(amp: float, freq: float, duration_sec: float) -> npt.NDArray[np.float32]:
    t = np.arange(int(SAMPLE_RATE * duration_sec)) / SAMPLE_RATE    # range between 0..duration
    y = amp * np.sin(2 * np.pi * freq * t)                          # if amp greater than 1 clipping will occur
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)


if __name__ == "__main__":
    f = generate_sin_array_normalized(5, 73.42, 2)
    write_wave("audio/sin_73_42.wav", f)
