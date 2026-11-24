import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile

import librosa
import mir_eval
import numpy as np
import numpy.typing as npt
import soundfile as sf

from soche.chords import PITCH_CLASSES, get_chords_file

ZENODO_RECORD_URL = "https://zenodo.org/records/3371780/files"
VOCALSET_API_URL = "https://zenodo.org/api/records/1442513"
GROOVE_ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"
MIX_SR = 44_100
GUITARSETPLUS_MIXES: tuple[str, ...] = ("vocals", "drums", "both")

GUITARSET_ARCHIVES: tuple[tuple[str, str, str], ...] = (
    ("annotation.zip", "annotation", "*.jams"),
    ("audio_mono-mic.zip", "audio_mono-mic", "*.wav"),
)
METRIC_KEYS = ("root", "majmin", "sevenths", "mirex", "overseg", "underseg", "seg")
STYLE_ORDER = ("comp", "solo", "vocals", "drums", "both")

_MAJ_SEMITONES = np.array(mir_eval.chord.QUALITIES["maj"][:8])
_MIN_SEMITONES = np.array(mir_eval.chord.QUALITIES["min"][:8])


def _is_valid_audio_file(path: Path) -> bool:
    return not path.name.startswith("._") and "__MACOSX" not in path.parts


def _download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}...")
    req = urllib.request.Request(url, headers={"User-Agent": "soche-eval/0.1"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        next_report = 25 * 1024 * 1024
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    print(f"  {downloaded / (1024 * 1024):.1f} / {total_bytes / (1024 * 1024):.1f} MB ({pct:.0f}%)")
                else:
                    print(f"  {downloaded / (1024 * 1024):.1f} MB")
                next_report += 25 * 1024 * 1024


def _download_and_extract(url: str, target_dir: Path, archive_name: str, tmp_parent: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=archive_name, suffix=".zip", dir=tmp_parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _download_file(url, tmp_path)
        print(f"Extracting {archive_name} -> {target_dir}...")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(target_dir)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def download_guitarset(data_dir: Path) -> None:
    """Download and extract GuitarSet annotations and mono-mic audio into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for archive_name, subdir_name, marker_glob in GUITARSET_ARCHIVES:
        target_dir = data_dir / subdir_name
        if target_dir.exists() and any(target_dir.rglob(marker_glob)):
            print(f"Found existing {subdir_name} in {target_dir}, skipping download.")
            continue
        _download_and_extract(
            f"{ZENODO_RECORD_URL}/{archive_name}?download=1",
            target_dir,
            archive_name,
            data_dir,
        )


def _download_vocalset(sources_dir: Path) -> Path:
    vocal_dir = sources_dir / "vocalset"
    if vocal_dir.exists() and any(_is_valid_audio_file(p) for p in vocal_dir.rglob("*.wav")):
        print(f"Found existing VocalSet in {vocal_dir}, skipping download.")
        return vocal_dir

    req = urllib.request.Request(VOCALSET_API_URL, headers={"User-Agent": "soche-eval/0.1"})
    with urllib.request.urlopen(req) as resp:
        record = json.loads(resp.read().decode("utf-8"))
    zip_files = [f for f in record.get("files", []) if str(f.get("key", "")).lower().endswith(".zip")]
    if not zip_files:
        raise RuntimeError(f"No .zip archive found in VocalSet Zenodo record {VOCALSET_API_URL}")
    entry = max(zip_files, key=lambda f: int(f.get("size", 0)))
    links = entry.get("links", {})
    url = links.get("self") or links.get("download")
    if not url:
        raise RuntimeError(f"Missing download URL for VocalSet file {entry.get('key')}")

    _download_and_extract(url, vocal_dir, str(entry["key"]), sources_dir)
    return vocal_dir


def _download_groove(sources_dir: Path) -> Path:
    groove_dir = sources_dir / "groove"
    if (
        groove_dir.exists()
        and any(groove_dir.rglob("info.csv"))
        and any(_is_valid_audio_file(p) for p in groove_dir.rglob("*.wav"))
    ):
        print(f"Found existing Groove MIDI Dataset in {groove_dir}, skipping download.")
        return groove_dir

    _download_and_extract(GROOVE_ZIP_URL, groove_dir, "groove-v1.0.0.zip", sources_dir)
    return groove_dir


def _active_rms(x: npt.NDArray[np.floating]) -> float:
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return 0.0
    active = x[np.abs(x) > 0.01 * peak]
    target = active if active.size > 0 else x
    return float(np.sqrt(np.mean(target**2)))


def _crossfade_concat(chunks: list[npt.NDArray[np.float64]], fade_samples: int) -> npt.NDArray[np.float64]:
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        fade = min(fade_samples, len(out) // 2, len(nxt) // 2)
        if fade > 0:
            w = np.linspace(0.0, 1.0, fade, dtype=np.float64)
            blended = out[-fade:] * (1.0 - w) + nxt[:fade] * w
            out = np.concatenate([out[:-fade], blended, nxt[fade:]])
        else:
            out = np.concatenate([out, nxt])
    return out


def _fit_length_loop(
    x: npt.NDArray[np.float64],
    n_samples: int,
    fade_samples: int,
    offset: int = 0,
) -> npt.NDArray[np.float64]:
    if x.size == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if len(x) >= n_samples:
        slack = len(x) - n_samples
        start = offset % (slack + 1) if slack > 0 else 0
        return x[start : start + n_samples]

    reps = (n_samples // max(1, len(x) - fade_samples)) + 2
    looped = _crossfade_concat([x] * reps, fade_samples)
    return looped[:n_samples]


def _normalize_peak(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (0.95 / peak) if peak > 0.98 else x


def _collect_vocal_pool(vocal_dir: Path) -> list[Path]:
    all_wavs = sorted(p for p in vocal_dir.rglob("*.wav") if _is_valid_audio_file(p))
    if not all_wavs:
        raise FileNotFoundError(f"No VocalSet WAV files found under {vocal_dir}")
    preferred = [
        p for p in all_wavs if any(part in p.parts for part in ("excerpts", "arpeggios", "scales"))
    ]
    return preferred if preferred else all_wavs


def _collect_groove_beats(groove_dir: Path) -> list[tuple[Path, float]]:
    info_csv = next(groove_dir.rglob("info.csv"), None)
    if info_csv is None:
        raise FileNotFoundError(f"No info.csv found under {groove_dir}")

    base_dir = info_csv.parent
    beats: list[tuple[Path, float]] = []
    with info_csv.open("r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rel = (row.get("audio_filename") or "").strip()
            if not rel or row.get("beat_type") != "beat":
                continue
            try:
                bpm = float(row["bpm"])
                dur = float(row["duration"])
            except (KeyError, TypeError, ValueError):
                continue
            if dur < 5.0 or bpm <= 0.0:
                continue
            wav_path = base_dir / rel
            if wav_path.is_file():
                beats.append((wav_path, bpm))

    if not beats:
        raise RuntimeError(f"No valid drum beat audio entries found in {info_csv}")
    return beats


def _mix_one_track(task: dict[str, Any]) -> dict[str, Any]:
    stem: str = task["stem"]
    jams_path: Path = task["jams_path"]
    guitar_wav: Path = task["guitar_wav"]
    drum_wav: Path = task["drum_wav"]
    vocal_wavs: list[Path] = task["vocal_wavs"]
    plus_dir: Path = task["plus_dir"]
    drum_offset: int = task["drum_offset"]

    fade_samples = int(0.05 * MIX_SR)
    guitar, _ = librosa.load(guitar_wav, sr=MIX_SR, mono=True)
    guitar = np.asarray(guitar, dtype=np.float64)
    n_samples = len(guitar)
    g_rms = _active_rms(guitar)

    drum_raw, _ = librosa.load(drum_wav, sr=MIX_SR, mono=True)
    drum_trim, _ = librosa.effects.trim(np.asarray(drum_raw, dtype=np.float64), top_db=35)
    if drum_trim.size == 0:
        drum_trim = np.asarray(drum_raw, dtype=np.float64)
    drums = _fit_length_loop(drum_trim, n_samples, fade_samples, offset=drum_offset)
    d_rms = _active_rms(drums)
    drums_scaled = drums * (g_rms / d_rms) if d_rms > 1e-12 else drums

    vocal_chunks: list[npt.NDArray[np.float64]] = []
    total_v = 0
    used_vocals: list[str] = []
    idx = 0
    while total_v < n_samples + fade_samples:
        v_path = vocal_wavs[idx % len(vocal_wavs)]
        v_raw, _ = librosa.load(v_path, sr=MIX_SR, mono=True)
        v_trim, _ = librosa.effects.trim(np.asarray(v_raw, dtype=np.float64), top_db=30)
        if v_trim.size == 0:
            v_trim = np.asarray(v_raw, dtype=np.float64)
        vocal_chunks.append(v_trim)
        used_vocals.append(v_path.name)
        total_v += max(1, len(v_trim) - fade_samples)
        idx += 1

    vocals = _crossfade_concat(vocal_chunks, fade_samples)[:n_samples]
    v_rms = _active_rms(vocals)
    vocals_scaled = vocals * (g_rms / v_rms) if v_rms > 1e-12 else vocals

    mixes = {
        "vocals": _normalize_peak(guitar + vocals_scaled),
        "drums": _normalize_peak(guitar + drums_scaled),
        "both": _normalize_peak(guitar + vocals_scaled + drums_scaled),
    }
    for mix_name, audio in mixes.items():
        out_path = plus_dir / mix_name / f"{stem}_mic.wav"
        sf.write(out_path, audio, MIX_SR, subtype="PCM_16")

    shutil.copy2(jams_path, plus_dir / "annotation" / f"{stem}.jams")
    return {
        "track_id": stem,
        "guitar_bpm": task["guitar_bpm"],
        "drum_file": drum_wav.name,
        "drum_bpm": task["drum_bpm"],
        "vocal_files": used_vocals,
        "snr_db": 0.0,
    }


def build_guitarsetplus(
    guitarset_dir: Path,
    plus_dir: Path,
    seed: int = 42,
    workers: int = os.cpu_count() or 1,
) -> None:
    """Download VocalSet + Groove MIDI and build the guitarsetplus mixed benchmark."""
    download_guitarset(guitarset_dir)
    pairs = find_guitarset_pairs(guitarset_dir, split="comp")
    expected = len(pairs)

    manifest_path = plus_dir / "manifest.json"
    ann_dir = plus_dir / "annotation"
    if (
        manifest_path.is_file()
        and ann_dir.is_dir()
        and len(list(ann_dir.glob("*.jams"))) == expected
        and all(len(list((plus_dir / m).glob("*.wav"))) == expected for m in GUITARSETPLUS_MIXES)
    ):
        print(f"Found existing guitarsetplus ({expected} x 3 mixes) in {plus_dir}, skipping build.")
        return

    sources_dir = plus_dir / "_sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    vocal_dir = _download_vocalset(sources_dir)
    groove_dir = _download_groove(sources_dir)

    vocal_pool = _collect_vocal_pool(vocal_dir)
    groove_beats = _collect_groove_beats(groove_dir)

    rng = random.Random(seed)
    shuffled_vocals = vocal_pool.copy()
    rng.shuffle(shuffled_vocals)

    ann_dir.mkdir(parents=True, exist_ok=True)
    for m in GUITARSETPLUS_MIXES:
        (plus_dir / m).mkdir(parents=True, exist_ok=True)

    drum_usage: Counter[Path] = Counter()
    tasks: list[dict[str, Any]] = []
    v_cursor = 0
    for stem, jams_path, wav_path in pairs:
        bpm_match = re.search(r"-(\d+)-", stem)
        target_bpm = float(bpm_match.group(1)) if bpm_match else 120.0

        cands = [(p, b) for p, b in groove_beats if abs(b - target_bpm) <= 5.0]
        if len(cands) < 3:
            cands = [(p, b) for p, b in groove_beats if abs(b - target_bpm) <= 10.0]
        if not cands:
            cands = sorted(groove_beats, key=lambda item: abs(item[1] - target_bpm))[:10]

        min_use = min(drum_usage[p] for p, _ in cands)
        least_used = [(p, b) for p, b in cands if drum_usage[p] == min_use]
        drum_wav, drum_bpm = rng.choice(least_used)
        drum_usage[drum_wav] += 1

        # Supply 8 shuffled vocal clips per track so trimmed speech/singing covers >30s.
        v_slice = [
            shuffled_vocals[(v_cursor + k) % len(shuffled_vocals)]
            for k in range(8)
        ]
        v_cursor += 8

        tasks.append(
            {
                "stem": stem,
                "jams_path": jams_path,
                "guitar_wav": wav_path,
                "guitar_bpm": target_bpm,
                "drum_wav": drum_wav,
                "drum_bpm": drum_bpm,
                "drum_offset": rng.randint(0, 10 * MIX_SR),
                "vocal_wavs": v_slice,
                "plus_dir": plus_dir,
            }
        )

    print(f"Mixing {len(tasks)} guitarsetplus tracks x {len(GUITARSETPLUS_MIXES)} variants...")
    if workers <= 1 or len(tasks) <= 1:
        entries = [_mix_one_track(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(_mix_one_track, tasks))

    manifest = {
        "dataset": "guitarsetplus",
        "seed": seed,
        "sample_rate": MIX_SR,
        "snr_db": 0.0,
        "sources": {
            "guitar": "GuitarSet comp (CC BY 4.0)",
            "vocals": "VocalSet (CC BY 4.0, Zenodo 1442513)",
            "drums": "Magenta Groove MIDI Dataset (CC BY 4.0)",
        },
        "tracks": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built guitarsetplus in {plus_dir} ({len(entries)} tracks x 3 mixes).")


def soche_to_harte(label: str) -> str:
    """Convert a SOCHE chord label ('C', 'Am', 'G7', 'N') to Harte notation."""
    if label == "N":
        return "N"
    if label.endswith("m"):
        return f"{label[:-1]}:min"
    if label.endswith("7"):
        return f"{label[:-1]}:7"
    return f"{label}:maj"


def load_guitarset_chords(
    jams_path: Path,
    source: str = "performed",
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Load chord (intervals, labels) from a GuitarSet JAMS file."""
    with jams_path.open("r", encoding="utf-8") as fp:
        jam = json.load(fp)

    chord_anns = [ann for ann in jam.get("annotations", []) if ann.get("namespace") == "chord"]
    if not chord_anns:
        raise ValueError(f"No chord annotations found in {jams_path}")

    if source == "performed":
        selected = next(
            (ann for ann in chord_anns if ann.get("annotation_metadata", {}).get("data_source")),
            chord_anns[-1],
        )
    else:
        selected = next(
            (ann for ann in chord_anns if not ann.get("annotation_metadata", {}).get("data_source")),
            chord_anns[0],
        )

    intervals: list[list[float]] = []
    labels: list[str] = []
    for obs in selected.get("data", []):
        time = float(obs["time"])
        duration = float(obs["duration"])
        if duration <= 0.0:
            continue
        intervals.append([time, time + duration])
        labels.append(str(obs["value"]))

    if not intervals:
        raise ValueError(f"No positive-duration chord intervals in {jams_path}")

    return np.asarray(intervals, dtype=np.float64), labels


def find_guitarset_pairs(
    data_dir: Path,
    split: str = "all",
) -> list[tuple[str, Path, Path]]:
    """Pair GuitarSet JAMS files with their corresponding mono-mic WAV files."""
    wav_by_stem = {p.stem: p for p in data_dir.rglob("*.wav")}
    pairs: list[tuple[str, Path, Path]] = []
    for jams_path in sorted(data_dir.rglob("*.jams")):
        stem = jams_path.stem
        if split == "comp" and not stem.endswith("_comp"):
            continue
        if split == "solo" and not stem.endswith("_solo"):
            continue

        wav_path = wav_by_stem.get(f"{stem}_mic") or wav_by_stem.get(stem)
        if wav_path is None:
            raise FileNotFoundError(f"Missing audio file for {jams_path}")
        pairs.append((stem, jams_path, wav_path))

    return pairs


def find_guitarsetplus_pairs(
    data_dir: Path,
    mix: str = "all",
) -> list[tuple[str, str, Path, Path]]:
    """Pair guitarsetplus JAMS files with their vocals/drums/both mixed WAV files."""
    ann_dir = data_dir / "annotation"
    if not ann_dir.is_dir():
        return []

    mix_names = GUITARSETPLUS_MIXES if mix == "all" else (mix,)
    pairs: list[tuple[str, str, Path, Path]] = []
    for mix_name in mix_names:
        mix_dir = data_dir / mix_name
        for jams_path in sorted(ann_dir.glob("*.jams")):
            stem = jams_path.stem
            wav_path = mix_dir / f"{stem}_mic.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(f"Missing mixed audio file {wav_path} for {jams_path}")
            pairs.append((f"{mix_name}/{stem}", mix_name, jams_path, wav_path))

    return pairs


def _canonical_majmin_label(label: str) -> str:
    """Normalize a Harte chord label to canonical maj/min/N (falling back to original)."""
    if label == "N":
        return "N"
    if label == "X":
        return "X"
    try:
        root, semitones, _ = mir_eval.chord.encode(label, reduce_extended_chords=False)
    except Exception:
        return label

    if root < 0 and np.all(semitones == 0):
        return "N"
    if np.all(semitones[:8] == _MAJ_SEMITONES):
        return f"{PITCH_CLASSES[int(root)]}:maj"
    if np.all(semitones[:8] == _MIN_SEMITONES):
        return f"{PITCH_CLASSES[int(root)]}:min"
    return label


def _filter_supported_label(label: str) -> str:
    """Keep only chords in SOCHE's vocabulary (maj, min, 7, N); mask others as 'X'."""
    if label in ("N", "X"):
        return label
    try:
        root, qual, ext, _ = mir_eval.chord.split(label)
    except Exception:
        return "X"
    if not ext and qual in ("maj", "min", "7"):
        return f"{root}:{qual}"
    return "X"


def evaluate_track(task: tuple[str, str | None, Path, Path, str, bool]) -> dict[str, Any]:
    """Run SOCHE chord extraction and mir_eval scoring on a single track."""
    track_id, style_override, jams_path, wav_path, chord_source, supported_only = task
    ref_intervals, ref_labels = load_guitarset_chords(jams_path, source=chord_source)
    if supported_only:
        ref_labels = [_filter_supported_label(lbl) for lbl in ref_labels]
    ref_end = float(ref_intervals[-1, 1])

    segments = get_chords_file(wav_path)
    valid_segments = [(float(s), float(e), c) for s, e, c in segments if e > s]
    if valid_segments:
        est_intervals = np.asarray([[s, e] for s, e, _ in valid_segments], dtype=np.float64)
        est_labels = [soche_to_harte(c) for _, _, c in valid_segments]
    else:
        est_intervals = np.asarray([[0.0, ref_end]], dtype=np.float64)
        est_labels = ["N"]

    if supported_only:
        eval_dur = float(
            sum(
                float(e - s)
                for (s, e), lbl in zip(ref_intervals, ref_labels)
                if lbl != "X"
            )
        )
    else:
        eval_dur = float(np.sum(ref_intervals[:, 1] - ref_intervals[:, 0]))

    if eval_dur > 0.0:
        scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
    else:
        scores = {k: 0.0 for k in METRIC_KEYS}

    adj_intervals, adj_labels = mir_eval.util.adjust_intervals(
        est_intervals,
        est_labels,
        float(ref_intervals.min()),
        float(ref_intervals.max()),
        mir_eval.chord.NO_CHORD,
        mir_eval.chord.NO_CHORD,
    )
    merged_intervals, merged_ref, merged_est = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, adj_intervals, adj_labels
    )
    durations = mir_eval.util.intervals_to_durations(merged_intervals)

    confusions: dict[tuple[str, str], float] = defaultdict(float)
    for ref_lbl, est_lbl, dur in zip(merged_ref, merged_est, durations):
        canon_ref = _canonical_majmin_label(ref_lbl)
        canon_est = _canonical_majmin_label(est_lbl)
        if canon_ref == "X" or canon_ref == canon_est:
            continue
        confusions[(canon_ref, canon_est)] += float(dur)

    if style_override is not None:
        style = style_override
    elif track_id.endswith("_comp"):
        style = "comp"
    elif track_id.endswith("_solo"):
        style = "solo"
    else:
        style = "other"

    return {
        "track_id": track_id,
        "style": style,
        "duration": eval_dur,
        "scores": {k: float(scores[k]) for k in METRIC_KEYS},
        "confusions": dict(confusions),
    }


def _weighted_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    total_dur = sum(r["duration"] for r in results)
    if total_dur <= 0.0:
        return {k: 0.0 for k in METRIC_KEYS}
    return {
        k: sum(r["scores"][k] * r["duration"] for r in results) / total_dur
        for k in METRIC_KEYS
    }


def _print_report(results: list[dict[str, Any]]) -> None:
    subsets = [("Overall", results)]
    for style in STYLE_ORDER:
        subset = [r for r in results if r["style"] == style]
        if subset:
            subsets.append((style, subset))

    header = f"{'Subset':<10} {'Tracks':>6} {'Dur(s)':>8} " + " ".join(
        f"{k:>8}" for k in METRIC_KEYS
    )
    print("\n" + header)
    print("-" * len(header))
    for name, subset in subsets:
        summary = _weighted_summary(subset)
        dur = sum(r["duration"] for r in subset)
        n_active = sum(1 for r in subset if r["duration"] > 0.0)
        cols = " ".join(f"{summary[k]:8.4f}" for k in METRIC_KEYS)
        print(f"{name:<10} {n_active:>6} {dur:>8.1f} {cols}")

    total_dur = sum(r["duration"] for r in results)
    merged_confusions: dict[tuple[str, str], float] = defaultdict(float)
    for r in results:
        for pair, dur in r["confusions"].items():
            merged_confusions[pair] += dur

    top_conf = sorted(merged_confusions.items(), key=lambda item: item[1], reverse=True)[:10]
    if top_conf:
        print("\nTop 10 confusion pairs (maj/min normalized):")
        print(f"{'Reference':<14} {'Estimated':<14} {'Dur(s)':>9} {'% Total':>8}")
        print("-" * 48)
        for (ref_lbl, est_lbl), dur in top_conf:
            pct = 100.0 * dur / total_dur if total_dur > 0 else 0.0
            print(f"{ref_lbl:<14} {est_lbl:<14} {dur:>9.1f} {pct:>7.2f}%")

    comp_results = [r for r in results if r["style"] == "comp" and r["duration"] > 0.0]
    worst_source = comp_results if comp_results else [r for r in results if r["duration"] > 0.0]
    worst_label = "comp" if comp_results else "overall"
    worst = sorted(worst_source, key=lambda r: r["scores"]["majmin"])[:5]
    if worst:
        print(f"\nWorst 5 {worst_label} tracks (by majmin):")
        print(f"{'Track':<28} {'majmin':>8} {'root':>8} {'sevenths':>8} {'seg':>8}")
        print("-" * 64)
        for r in worst:
            s = r["scores"]
            print(
                f"{r['track_id']:<28} {s['majmin']:>8.4f} {s['root']:>8.4f} "
                f"{s['sevenths']:>8.4f} {s['seg']:>8.4f}"
            )


def _save_benchmark(
    save_path: Path,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    subsets = {"overall": results}
    for style in STYLE_ORDER:
        subset = [r for r in results if r["style"] == style]
        if subset:
            subsets[style] = subset

    payload: dict[str, Any] = {
        "dataset": args.dataset,
        "chord_source": args.chord_source,
        "split": args.mix if args.dataset == "guitarsetplus" else args.split,
        "supported_only": args.supported_only,
        "subsets": {
            name: {
                "tracks": sum(1 for r in subset if r["duration"] > 0.0),
                "duration_sec": round(sum(r["duration"] for r in subset), 2),
                "metrics": {k: round(v, 4) for k, v in _weighted_summary(subset).items()},
            }
            for name, subset in subsets.items()
        },
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SOCHE chord extraction on GuitarSet / GuitarSetPlus.")
    parser.add_argument("--dataset", choices=("guitarset", "guitarsetplus"), default="guitarset")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--guitarset-dir", type=Path, default=Path("audio/guitarset"))
    parser.add_argument("--download", action="store_true", help="Download/build dataset if missing.")
    parser.add_argument("--download-only", action="store_true", help="Download/build dataset and exit.")
    parser.add_argument("--split", choices=("all", "comp", "solo"), default="all")
    parser.add_argument("--mix", choices=("all", "vocals", "drums", "both"), default="all")
    parser.add_argument("--chord-source", choices=("performed", "instructed"), default="performed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--supported-only",
        action="store_true",
        help="Mask reference chords outside SOCHE's vocabulary (maj, min, 7, N) as 'X'.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Save summary JSON to PATH.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N tracks.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    data_dir = args.data_dir or (
        Path("audio/guitarsetplus") if args.dataset == "guitarsetplus" else Path("audio/guitarset")
    )

    if args.download:
        if args.dataset == "guitarsetplus":
            build_guitarsetplus(args.guitarset_dir, data_dir, seed=args.seed, workers=args.workers)
        else:
            download_guitarset(data_dir)
        if args.download_only:
            return

    if args.dataset == "guitarsetplus":
        plus_pairs = find_guitarsetplus_pairs(data_dir, mix=args.mix)
        if not plus_pairs:
            raise SystemExit(
                f"No guitarsetplus tracks found in {data_dir}. Run with --dataset guitarsetplus --download first."
            )
        if args.limit is not None:
            plus_pairs = plus_pairs[: max(0, args.limit)]
        tasks = [
            (tid, mix_name, jams_p, wav_p, args.chord_source, args.supported_only)
            for tid, mix_name, jams_p, wav_p in plus_pairs
        ]
        scope_tag = f"mix={args.mix}"
    else:
        pairs = find_guitarset_pairs(data_dir, split=args.split)
        if not pairs:
            raise SystemExit(
                f"No GuitarSet tracks found in {data_dir}. Run with --download first."
            )
        if args.limit is not None:
            pairs = pairs[: max(0, args.limit)]
        tasks = [
            (tid, None, jams_p, wav_p, args.chord_source, args.supported_only)
            for tid, jams_p, wav_p in pairs
        ]
        scope_tag = f"split={args.split}"

    mode_tag = ", supported-only" if args.supported_only else ""
    print(f"Evaluating {len(tasks)} {args.dataset} tracks ({args.chord_source}, {scope_tag}{mode_tag})...")

    if args.workers <= 1 or len(tasks) <= 1:
        results = [evaluate_track(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate_track, tasks))

    _print_report(results)
    if args.save is not None:
        _save_benchmark(args.save, results, args)


if __name__ == "__main__":
    main()
