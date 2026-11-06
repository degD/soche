
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from soche.wav.constants import MAXVAL


def info_wave(wave_path: str) -> None:
    with wave.open(wave_path, "rb") as fp:
        _channels = fp.getnchannels()
        _sample_width = fp.getsampwidth()    # sample size, number of bytes required to represent a single sample
        _framerate = fp.getframerate()       # number of samples / seconds
        _num_frames = fp.getnframes()        # total number of samples in audio
        print(fp.getparams())


def read_wave(wave_path: str | Path) -> npt.NDArray[np.float32]:
    with wave.open(str(wave_path), "rb") as fp:
        num_of_frames = fp.getnframes();
        buffer = fp.readframes(num_of_frames)   # TODO: parse loop in fixed sizes
        data = np.frombuffer(buffer, "<i2")     # wav is little endian
        data = data.astype(np.float32)          # float32 is smallest possible size
        return data / MAXVAL                    # simpler way to normalize -1..1


if __name__ == "__main__":
    info_wave("audio/audio.wav")
    print(read_wave("audio/audio.wav"))
