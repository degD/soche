from pathlib import Path

from soche.algorithms.librosa_pyin import SocheAlgorithmPyin
from soche.runner import Soche


def main() -> None:

    algo = SocheAlgorithmPyin()
    soche = Soche(algo, 1000, Path("audio/audio.wav"))
    soche.process()
