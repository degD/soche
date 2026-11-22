
import numpy as np
import numpy.typing as npt

from soche.chords.match import (
    cosine_similarity_frame_and_chord,
    get_chords_list,
    plot_chord_matching,
)
from soche.fft.filter import harmony_median_filter
from soche.fft.stft import calc_chroma, stft
from soche.wav.read_wave import read_wave


def cosine_similarity_matrix(chroma: npt.NDArray) -> npt.NDArray:
    frame_chord_scores = []
    for frame in chroma.T:
        scores = []
        for chord in get_chords_list():
            s = cosine_similarity_frame_and_chord(frame, chord)
            scores.append(s)
        frame_chord_scores.append(scores)
    return np.array(frame_chord_scores)


def match_chords_viterbi(scores: npt.NDArray, switch_penalty: float) -> npt.NDArray:
    num_of_frames, num_of_chords = scores.shape

    # Build a matrix showing transition penalty from chord to chord.
    # Staying on the same chord has no cost. Switching costs a bit.
    transitions_penalty_matrix = np.full((num_of_chords, num_of_chords), -1 * switch_penalty)
    np.fill_diagonal(transitions_penalty_matrix, 0)

    cumulative_scores = np.zeros((num_of_frames, num_of_chords))
    backpointers = np.zeros((num_of_frames, num_of_chords), int)

    cumulative_scores[0] = scores[0]
    backpointers[0] = -1

    # Calculate cumulative scores and back pointers for tracing the path backwards.
    # Actually a dynamic programming approach for generating a matrix of best chords
    # to choose for each frame, considering past choices.
    for t in range(1, num_of_frames):

        # Candidates has shape (num_of_chords, num_of_chords).
        # Shows score of moving from a chord to another at that frame.
        candidates = cumulative_scores[t - 1][:, None] + transitions_penalty_matrix

        # Find the best candidate for each current chord choice.
        # Size is (1, num_of_chords).
        backpointers[t] = np.argmax(candidates, axis=0)

        # Best cumulative score at the current frame, for each possible
        # chord at that frame.
        cumulative_scores[t] = scores[t] + np.max(candidates, axis=0)

    # Trace the best path backward.
    chord_id = np.argmax(cumulative_scores[-1])
    path = [chord_id]
    for t in range(num_of_frames-1, 0, -1):
        chord_id = backpointers[t, chord_id]
        path.append(chord_id)

    return np.array(path[::-1])


def build_chord_matching_list(scores: npt.NDArray, path: npt.NDArray) -> list[tuple[int, float]]:
    chord_matching_list = []
    for t, chord_idx in enumerate(path):
        chord_matching_list.append((chord_idx, scores[t, chord_idx]))
    return chord_matching_list

if __name__ == "__main__":
    window_size = 4096
    hop_size = 512
    ref_freq = 440

    f = read_wave("audio/rec1.wav")
    stft_f = stft(f, window_size, hop_size)
    filtered_f = harmony_median_filter(stft_f, 15)
    chroma_f = calc_chroma(filtered_f, window_size, ref_freq)

    scores = cosine_similarity_matrix(chroma_f)
    path = match_chords_viterbi(scores, switch_penalty=0.6)
    chord_matching_list = build_chord_matching_list(scores, path)

    plot_chord_matching(chord_matching_list)
