
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt


def write_wave_int16(wave_path: str, array: npt.NDArray[np.int16]) -> None:
    with wave.open(wave_path, "wb") as fp:
        fp.setparams((1, 2, 16000, len(array), "NONE", "not compressed"))
        fp.writeframes(array.tobytes())


def de_normalize_float_array(array_normalized: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
    max_val = np.iinfo(np.int16).max
    min_val = np.iinfo(np.int16).min
    array_intermediate = (max_val - min_val) * (array_normalized + 1) / 2 + min_val
    return np.astype(np.round(array_intermediate), np.int16)    # rounding is important as cast does not round


def write_wave(wave_path: str | Path, array_normalized: npt.NDArray[np.float32]) -> None:
    array_int16 = de_normalize_float_array(array_normalized)
    write_wave_int16(str(wave_path), array_int16)


def generate_sin_array_normalized(amp: float, freq: float, duration_sec: float) -> npt.NDArray[np.float32]:
    x = np.linspace(0, duration_sec, int(16000 * duration_sec))
    y = amp * np.sin(2 * np.pi * freq * x)
    normalized = 2 * (y + amp) / (2 * amp) - 1
    return np.astype(normalized, np.float32)


if __name__ == "__main__":
    f = generate_sin_array_normalized(5, 73.42, 2)
    write_wave("audio/sin_73_42.wav", f)
