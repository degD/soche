from pathlib import Path

from soche.chords.match import get_chords_list
from soche.chords.viterbi import (
    build_chord_matching_list,
    cosine_similarity_matrix,
    match_chords_viterbi,
)
from soche.fft.filter import harmony_median_filter
from soche.fft.stft import calc_chroma, stft
from soche.wav.read_wave import read_wave


def get_chords_file(filename: str | Path) -> list[tuple[int, str]]:
    window_size = 4096
    hop_size = 512
    ref_freq = 440

    f = read_wave(filename)
    stft_f = stft(f, window_size, hop_size)
    filtered_f = harmony_median_filter(stft_f, 15)
    chroma_f = calc_chroma(filtered_f, window_size, ref_freq)

    scores = cosine_similarity_matrix(chroma_f)
    path = match_chords_viterbi(scores, switch_penalty=0.6)
    chord_matching_list = build_chord_matching_list(scores, path)

    frame_chords_list = []
    for chord_idx, _ in chord_matching_list:
        chord_lbl = get_chords_list()[chord_idx]
        frame_chords_list.append(chord_lbl)

    count = 1
    chords_and_counts = []
    prev_chord = frame_chords_list[0]
    for chord in frame_chords_list[1:]:
        if chord == prev_chord:
            count += 1
        else:
            chords_and_counts.append((count, prev_chord))
            count = 1
            prev_chord = chord

    return chords_and_counts
