from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
HOP_LENGTH = 512

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


def estimate_chroma(
    y: npt.NDArray[np.floating],
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> npt.NDArray[np.float64]:
    """Extract a harmonic Constant-Q chromagram of shape (12, n_frames)."""
    y_harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
    return np.asarray(chroma, dtype=np.float64)


def compute_chord_probabilities(
    chroma: npt.NDArray[np.floating],
    templates: npt.NDArray[np.float64],
    no_chord_index: int = -1,
    temperature: float = 0.1,
) -> npt.NDArray[np.float64]:
    """Compute per-frame chord probabilities of shape (n_chords, n_frames)."""
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    silent_frames = norms[0] < 1e-6
    chroma_norm = chroma / np.maximum(norms, 1e-12)

    # Cosine similarity between normalized templates and chroma frames.
    scores = templates @ chroma_norm
    if np.any(silent_frames):
        scores[:, silent_frames] = 0.0
        scores[no_chord_index, silent_frames] = 1.0

    scaled = scores / temperature
    scaled -= np.max(scaled, axis=0, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)


def decode_chords(
    probs: npt.NDArray[np.float64],
    labels: list[str],
    self_loop_prob: float = 0.9,
) -> list[str]:
    """Decode the most likely chord sequence using librosa's discriminative Viterbi."""
    transition = librosa.sequence.transition_loop(len(labels), self_loop_prob)
    path = librosa.sequence.viterbi_discriminative(probs, transition)
    return [labels[int(idx)] for idx in path]


def run_length_encode(chords: list[str]) -> list[tuple[int, str]]:
    """Collapse consecutive identical chord labels into (frame_count, label) pairs."""
    if not chords:
        return []

    encoded: list[tuple[int, str]] = []
    prev_chord = chords[0]
    count = 1
    for chord in chords[1:]:
        if chord == prev_chord:
            count += 1
        else:
            encoded.append((count, prev_chord))
            prev_chord = chord
            count = 1
    encoded.append((count, prev_chord))
    return encoded


def get_chords_file(
    filename: str | Path,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    self_loop_prob: float = 0.9,
) -> list[tuple[int, str]]:
    """Estimate chords from an audio file and return run-length encoded (count, chord) pairs."""
    y, loaded_sr = librosa.load(filename, sr=sr, mono=True)
    if y.size == 0:
        return []

    chroma = estimate_chroma(y, sr=int(loaded_sr), hop_length=hop_length)
    labels, templates = build_chord_templates()
    no_chord_index = labels.index("N")
    probs = compute_chord_probabilities(chroma, templates, no_chord_index=no_chord_index)
    frame_chords = decode_chords(probs, labels, self_loop_prob=self_loop_prob)
    return run_length_encode(frame_chords)
