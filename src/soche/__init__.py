from pathlib import Path

from soche.algorithms.demo import SocheAlgorithmDemo
from soche.runner import Soche


def main() -> None:

    algo = SocheAlgorithmDemo()
    soche = Soche(algo, 1000, Path("audio/audio.wav"))
    soche.process()
