
import numpy as np

from soche.runner import SocheAlgorithmInterface, SocheChordClasses


class SocheAlgorithmDemo(SocheAlgorithmInterface):

    def __init__(self):
        pass

    def processSample(self, sample: np.ndarray):
        print("Hello world! (A)")
        return SocheChordClasses.A
