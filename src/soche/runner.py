
import subprocess as sp
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def convertSong(inputPath: Path, outPath: Path) -> None:
    sp.run([
        "ffmpeg", "-i", inputPath, "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000", outPath
    ], check=False)


class SocheChordClasses(Enum):
    C   = 1,
    D_B = 2,
    D   = 3,
    E_B = 4,
    E   = 5,
    F   = 6,
    G_B = 7,
    G   = 8,
    A_B = 9,
    A   = 10,
    B   = 11,
    B_B = 12,


class SocheAlgorithmInterface(ABC):

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def processSample(self, sample: np.ndarray) -> SocheChordClasses:
        pass


class Soche:

    def __init__(self,
        algorithm: SocheAlgorithmInterface,
        periodMs: int,
        songPath: Path
    ):
        self.algorithm = algorithm
        self.periodMs = periodMs
        self.songPath = songPath

    def process(self) -> list[SocheChordClasses]:
        sr, data = wavfile.read(self.songPath)
        chords = []
        for i in range(0, len(data), round(sr * self.periodMs / 1000)):
            sample = data[i:(i+sr)]
            c = self.algorithm.processSample(sample)
            chords.append(c)
        return chords
