import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile

import mir_eval
import numpy as np
import numpy.typing as npt

from soche.chords import PITCH_CLASSES, get_chords_file

ZENODO_RECORD_URL = "https://zenodo.org/records/3371780/files"
GUITARSET_ARCHIVES: tuple[tuple[str, str, str], ...] = (
    ("annotation.zip", "annotation", "*.jams"),
    ("audio_mono-mic.zip", "audio_mono-mic", "*.wav"),
)
METRIC_KEYS = ("root", "majmin", "sevenths", "mirex", "overseg", "underseg", "seg")

_MAJ_SEMITONES = np.array(mir_eval.chord.QUALITIES["maj"][:8])
_MIN_SEMITONES = np.array(mir_eval.chord.QUALITIES["min"][:8])


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


def download_guitarset(data_dir: Path) -> None:
    """Download and extract GuitarSet annotations and mono-mic audio into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for archive_name, subdir_name, marker_glob in GUITARSET_ARCHIVES:
        target_dir = data_dir / subdir_name
        if target_dir.exists() and any(target_dir.rglob(marker_glob)):
            print(f"Found existing {subdir_name} in {target_dir}, skipping download.")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=archive_name, suffix=".zip", dir=data_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            _download_file(f"{ZENODO_RECORD_URL}/{archive_name}?download=1", tmp_path)
            print(f"Extracting {archive_name} -> {target_dir}...")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(target_dir)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        finally:
            tmp_path.unlink(missing_ok=True)


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


def evaluate_track(task: tuple[str, Path, Path, str]) -> dict[str, Any]:
    """Run SOCHE chord extraction and mir_eval scoring on a single GuitarSet track."""
    track_id, jams_path, wav_path, chord_source = task
    ref_intervals, ref_labels = load_guitarset_chords(jams_path, source=chord_source)
    ref_end = float(ref_intervals[-1, 1])

    segments = get_chords_file(wav_path)
    valid_segments = [(float(s), float(e), c) for s, e, c in segments if e > s]
    if valid_segments:
        est_intervals = np.asarray([[s, e] for s, e, _ in valid_segments], dtype=np.float64)
        est_labels = [soche_to_harte(c) for _, _, c in valid_segments]
    else:
        est_intervals = np.asarray([[0.0, ref_end]], dtype=np.float64)
        est_labels = ["N"]

    scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)

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

    if track_id.endswith("_comp"):
        style = "comp"
    elif track_id.endswith("_solo"):
        style = "solo"
    else:
        style = "other"

    total_dur = float(np.sum(ref_intervals[:, 1] - ref_intervals[:, 0]))
    return {
        "track_id": track_id,
        "style": style,
        "duration": total_dur,
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
    for style in ("comp", "solo"):
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
        cols = " ".join(f"{summary[k]:8.4f}" for k in METRIC_KEYS)
        print(f"{name:<10} {len(subset):>6} {dur:>8.1f} {cols}")

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

    comp_results = [r for r in results if r["style"] == "comp"]
    worst_source = comp_results if comp_results else results
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SOCHE chord extraction on GuitarSet.")
    parser.add_argument("--data-dir", type=Path, default=Path("audio/guitarset"))
    parser.add_argument("--download", action="store_true", help="Download GuitarSet if missing.")
    parser.add_argument("--download-only", action="store_true", help="Download GuitarSet and exit.")
    parser.add_argument("--split", choices=("all", "comp", "solo"), default="all")
    parser.add_argument("--chord-source", choices=("performed", "instructed"), default="performed")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N tracks.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    if args.download:
        download_guitarset(args.data_dir)
        if args.download_only:
            return

    pairs = find_guitarset_pairs(args.data_dir, split=args.split)
    if not pairs:
        raise SystemExit(
            f"No GuitarSet tracks found in {args.data_dir}. Run with --download first."
        )
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit)]

    tasks = [(tid, jams_p, wav_p, args.chord_source) for tid, jams_p, wav_p in pairs]
    print(f"Evaluating {len(tasks)} GuitarSet tracks ({args.chord_source}, split={args.split})...")

    if args.workers <= 1 or len(tasks) <= 1:
        results = [evaluate_track(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate_track, tasks))

    _print_report(results)


if __name__ == "__main__":
    main()
