
from enum import Enum

import librosa
import numpy as np

from soche.runner import SocheAlgorithmInterface, SocheChordClasses


class NotesAndFrequencies(Enum):
    C4       = 261.63
    C_SHARP4 = 277.18
    D4       = 293.66
    D_SHARP4 = 311.13
    E4       = 329.63
    F4       = 349.23
    F_SHARP4 = 369.99
    G4       = 392.00
    G_SHARP4 = 415.30
    A4       = 440.00
    A_SHARP4 = 466.16
    B4       = 493.88
    C5       = 523.25
    C_SHARP5 = 554.37
    D5       = 587.33
    D_SHARP5 = 622.25
    E5       = 659.26
    F5       = 698.46
    F_SHARP5 = 739.99
    G5       = 783.99
    G_SHARP5 = 830.61
    A5       = 880.00
    A_SHARP5 = 932.33
    B5       = 987.77


def getClosestNote(n: float) -> NotesAndFrequencies:
    notes = []
    for c in NotesAndFrequencies:
        notes.append((c, abs(c.value - n)))
    v = min(notes, key=lambda c: c[1])[0]
    return v


def getMatchingChord(n: NotesAndFrequencies) -> SocheChordClasses:
    if "_" in n.name:
        if n.name[0] == 'C':
            return SocheChordClasses.D_B
        elif n.name[0] == 'D':
            return SocheChordClasses.E_B
        elif n.name[0] == 'F':
            return SocheChordClasses.G_B
        elif n.name[0] == 'G':
            return SocheChordClasses.A_B
        elif n.name[0] == 'A':
            return SocheChordClasses.B_B
        else:
            return SocheChordClasses.A

    else:
        if n.name[0] == 'C':
            return SocheChordClasses.C
        elif n.name[0] == 'D':
            return SocheChordClasses.D
        elif n.name[0] == 'E':
            return SocheChordClasses.E
        elif n.name[0] == 'F':
            return SocheChordClasses.F
        elif n.name[0] == 'G':
            return SocheChordClasses.G
        elif n.name[0] == 'A':
            return SocheChordClasses.A
        elif n.name[0] == 'B':
            return SocheChordClasses.B
        else:
            return SocheChordClasses.A


class SocheAlgorithmPyin(SocheAlgorithmInterface):

    def __init__(self):
        pass

    def processSample(self, sample: np.ndarray):
        pyin_f0, _, _ = librosa.pyin(y=sample, sr=16000, fmin=250, fmax=1000)
        mean = sum(y for y in pyin_f0) / len(pyin_f0)
        closest = getClosestNote(mean)

        print(f"For this sample: {closest.name}")
        return getMatchingChord(closest)
