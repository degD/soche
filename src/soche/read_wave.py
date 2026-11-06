
import wave

import numpy as np
import numpy.typing as npt


def info_wave(wave_path: str) -> None:
    with wave.open(wave_path, "rb") as fp:
        _channels = fp.getnchannels()
        _sample_width = fp.getsampwidth()    # sample size, number of bytes required to represent a single sample
        _framerate = fp.getframerate()       # number of samples / seconds
        _num_frames = fp.getnframes()        # total number of samples in audio
        print(fp.getparams())


def normalize_int16_array(array: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
    max_val = np.iinfo(np.int16).max
    min_val = np.iinfo(np.int16).min
    array32 = np.astype(array, np.int32)
    return 2 * (array32 - min_val) / (max_val - min_val) - 1


def read_wave(wave_path: str):
    with wave.open(wave_path, "rb") as fp:
        num_of_frames = fp.getnframes();
        buffer = fp.readframes(num_of_frames)   # TODO: parse loop in fixed sizes
        data = np.frombuffer(buffer, np.int16)  # TODO: is wav pcm little or big endian?
        print(normalize_int16_array(data))      # TODO: is float64 too much? would float16 enough?
        return normalize_int16_array(data)


if __name__ == "__main__":
    # info_wave("audio/audio.wav")
    read_wave("audio/audio.wav")
