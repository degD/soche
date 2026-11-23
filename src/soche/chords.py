from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
HOP_LENGTH = 512

CQT_N_OCTAVES = 5
CQT_BINS_PER_SEMITONE = 3
CQT_BINS_PER_OCTAVE = 12 * CQT_BINS_PER_SEMITONE
CQT_N_SEMITONES = 12 * CQT_N_OCTAVES
CQT_N_BINS = CQT_N_SEMITONES * CQT_BINS_PER_SEMITONE
# Offset down by 1/3 semitone so each semitone m is centered at bin 3*m + 1.
CQT_FMIN = float(librosa.note_to_hz("C1") * (2.0 ** (-1.0 / CQT_BINS_PER_OCTAVE)))

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


def extract_harmonic_cqt(
    y: npt.NDArray[np.floating],
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    hpss_margin: float = 2.5,
) -> npt.NDArray[np.float64]:
    """Compute harmonic semitone CQT magnitude of shape (60, n_frames) over C1-B5."""
    cqt_mag = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=CQT_FMIN,
            n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
            tuning=None,
        )
    )
    cqt_harm, _ = librosa.decompose.hpss(cqt_mag, margin=hpss_margin)
    # Pool each semitone's 3 sub-bins (m-1/3, m, m+1/3) via peak magnitude.
    semitone_cqt = cqt_harm.reshape(CQT_N_SEMITONES, CQT_BINS_PER_SEMITONE, -1).max(axis=1)
    return np.asarray(semitone_cqt, dtype=np.float64)


def compute_beat_boundaries(
    y: npt.NDArray[np.floating],
    n_frames: int,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    subdivisions: int = 2,
) -> npt.NDArray[np.int64]:
    """Return sorted frame boundaries [0, ..., n_frames] aligned to beats or sub-beats."""
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_frames = [int(b) for b in beats if 0 < int(b) < n_frames]

    if len(beat_frames) < 4:
        step = max(1, int(round((0.5 / subdivisions) * sr / hop_length)))
        grid = list(range(step, n_frames, step))
    elif subdivisions > 1:
        grid = list(beat_frames)
        for i in range(len(beat_frames) - 1):
            start, end = beat_frames[i], beat_frames[i + 1]
            for k in range(1, subdivisions):
                grid.append(start + (k * (end - start)) // subdivisions)
        grid.sort()
    else:
        grid = beat_frames

    return np.unique(np.concatenate([[0], grid, [n_frames]])).astype(np.int64)


def sync_to_boundaries(
    spec: npt.NDArray[np.float64],
    boundaries: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Aggregate frame-level features across beat/sub-beat spans via mean (arpeggio-safe)."""
    n_units = len(boundaries) - 1
    synced = np.empty((spec.shape[0], n_units), dtype=np.float64)
    for i in range(n_units):
        synced[:, i] = spec[:, boundaries[i] : boundaries[i + 1]].mean(axis=1)
    return synced


def separate_repeating_harmony(
    spec: npt.NDArray[np.float64],
    subdivisions: int = 2,
    margin_bg: float = 1.5,
) -> npt.NDArray[np.float64]:
    """Suppress non-repeating foreground melody via non-local median filtering."""
    n_units = spec.shape[1]
    min_units = 16 * subdivisions
    if n_units < min_units:
        return spec

    width = max(2, min(4 * subdivisions, n_units // 4))
    bg = librosa.decompose.nn_filter(
        spec,
        aggregate=np.median,
        metric="cosine",
        width=width,
    )
    bg = np.minimum(spec, bg)
    fg = spec - bg
    mask = librosa.util.softmask(bg, margin_bg * fg, power=2.0)
    return np.asarray(mask * spec, dtype=np.float64)


def fold_to_chroma(semitone_spec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fold a (12 * n_octaves, n_frames) pitch matrix into a (12, n_frames) chromagram."""
    n_octaves = semitone_spec.shape[0] // 12
    return semitone_spec[: 12 * n_octaves].reshape(n_octaves, 12, -1).sum(axis=0)


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
    self_loop_prob: float = 0.8,
) -> list[str]:
    """Decode the most likely chord sequence using librosa's discriminative Viterbi."""
    transition = librosa.sequence.transition_loop(len(labels), self_loop_prob)
    path = librosa.sequence.viterbi_discriminative(probs, transition)
    return [labels[int(idx)] for idx in path]


def segments_from_chords(
    chords: list[str],
    boundaries_sec: npt.NDArray[np.float64],
) -> list[tuple[float, float, str]]:
    """Collapse consecutive identical chords into (start_sec, end_sec, chord) tuples."""
    if not chords:
        return []

    segments: list[tuple[float, float, str]] = []
    start_idx = 0
    prev_chord = chords[0]
    for i in range(1, len(chords)):
        if chords[i] != prev_chord:
            segments.append((float(boundaries_sec[start_idx]), float(boundaries_sec[i]), prev_chord))
            start_idx = i
            prev_chord = chords[i]
    segments.append((float(boundaries_sec[start_idx]), float(boundaries_sec[len(chords)]), prev_chord))
    return segments


def get_chords_file(
    filename: str | Path,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    beat_subdivisions: int = 2,
    self_loop_prob: float = 0.8,
) -> list[tuple[float, float, str]]:
    """Estimate chords from an audio file and return (start_sec, end_sec, chord) segments."""
    y, loaded_sr = librosa.load(filename, sr=sr, mono=True)
    if y.size == 0:
        return []

    sr_int = int(loaded_sr)
    duration = float(y.size) / sr_int
    semitone_cqt = extract_harmonic_cqt(y, sr=sr_int, hop_length=hop_length)
    boundaries = compute_beat_boundaries(
        y,
        n_frames=semitone_cqt.shape[1],
        sr=sr_int,
        hop_length=hop_length,
        subdivisions=beat_subdivisions,
    )
    synced_cqt = sync_to_boundaries(semitone_cqt, boundaries)
    harmony_cqt = separate_repeating_harmony(synced_cqt, subdivisions=beat_subdivisions)
    chroma = fold_to_chroma(harmony_cqt)

    labels, templates = build_chord_templates()
    no_chord_index = labels.index("N")
    probs = compute_chord_probabilities(chroma, templates, no_chord_index=no_chord_index)
    unit_chords = decode_chords(probs, labels, self_loop_prob=self_loop_prob)

    boundaries_sec = librosa.frames_to_time(boundaries, sr=sr_int, hop_length=hop_length)
    boundaries_sec[-1] = duration
    return segments_from_chords(unit_chords, boundaries_sec)
