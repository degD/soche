import numpy as np
import numpy.typing as npt

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Relative weights for chord tones: root, 3rd, and 5th.
_CHORD_QUALITIES: dict[str, dict[int, float]] = {
    "": {0: 1.20, 4: 0.75, 7: 0.85},   # Major triad
    "m": {0: 1.20, 3: 0.75, 7: 0.85},  # Minor triad
}


def _build_templates_dict() -> dict[str, npt.NDArray[np.float64]]:
    templates: dict[str, npt.NDArray[np.float64]] = {}
    for suffix, intervals in _CHORD_QUALITIES.items():
        for root_idx, root_name in enumerate(PITCH_CLASSES):
            vec = np.zeros(12, dtype=np.float64)
            for interval, weight in intervals.items():
                vec[(root_idx + interval) % 12] = weight
            templates[f"{root_name}{suffix}"] = vec

    # No-chord template: uniform energy across all pitch classes.
    templates["N"] = np.ones(12, dtype=np.float64)
    return templates


CHORD_TEMPLATES: dict[str, npt.NDArray[np.float64]] = _build_templates_dict()


def build_chord_templates() -> tuple[list[str], npt.NDArray[np.float64]]:
    """Return chord labels and L2-normalized template matrix of shape (n_chords, 12)."""
    labels = list(CHORD_TEMPLATES.keys())
    matrix = np.vstack([CHORD_TEMPLATES[label] for label in labels])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return labels, matrix / np.maximum(norms, 1e-12)
