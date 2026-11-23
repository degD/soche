from pathlib import Path

import numpy as np
import soundfile as sf

from soche.chords.output import (
    HOP_LENGTH,
    SAMPLE_RATE,
    compute_chord_probabilities,
    get_chords_file,
    run_length_encode,
)
from soche.chords.templates import CHORD_TEMPLATES, build_chord_templates
from soche.player.player import app


def _synth_chord(freqs: list[float], duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (signal / max(len(freqs), 1) * 0.5).astype(np.float32)


def test_build_chord_templates_shape_and_norms() -> None:
    labels, matrix = build_chord_templates()
    assert len(labels) == 25
    assert "C" in labels
    assert "Am" in labels
    assert "N" in labels
    assert matrix.shape == (25, 12)
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    assert set(CHORD_TEMPLATES.keys()) == set(labels)


def test_run_length_encode() -> None:
    assert run_length_encode([]) == []
    assert run_length_encode(["C"]) == [(1, "C")]
    assert run_length_encode(["C", "C", "Am", "G", "G", "G"]) == [
        (2, "C"),
        (1, "Am"),
        (3, "G"),
    ]


def test_compute_chord_probabilities() -> None:
    labels, templates = build_chord_templates()
    no_chord_idx = labels.index("N")

    c_major_chroma = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0])[:, None]
    silent_chroma = np.zeros((12, 1))
    chroma = np.hstack([c_major_chroma, silent_chroma])

    probs = compute_chord_probabilities(chroma, templates, no_chord_index=no_chord_idx)
    assert probs.shape == (25, 2)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-12)
    assert labels[int(np.argmax(probs[:, 0]))] == "C"
    assert labels[int(np.argmax(probs[:, 1]))] == "N"


def test_get_chords_file_progression(tmp_path: Path) -> None:
    c_major = _synth_chord([261.63, 329.63, 392.00], duration=1.5)
    a_minor = _synth_chord([220.00, 261.63, 329.63], duration=1.5)
    silence = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
    audio = np.concatenate([c_major, a_minor, silence])

    wav_path = tmp_path / "progression.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    result = get_chords_file(wav_path)
    assert len(result) > 0
    total_frames = sum(count for count, _ in result)
    expected_frames = 1 + len(audio) // HOP_LENGTH
    assert abs(total_frames - expected_frames) <= 2

    chords = [chord for _, chord in result]
    assert "C" in chords
    assert "Am" in chords
    assert chords[-1] == "N"


def test_player_get_chords_endpoint(tmp_path: Path) -> None:
    c_major = _synth_chord([261.63, 329.63, 392.00], duration=0.8)
    wav_path = tmp_path / "c_major.wav"
    sf.write(wav_path, c_major, SAMPLE_RATE)

    client = app.test_client()
    response = client.post("/get_chords", data=wav_path.read_bytes())
    assert response.status_code == 200
    payload = response.get_json(force=True)
    assert isinstance(payload, list)
    assert len(payload) > 0
    assert any(chord == "C" for _, chord in payload)
