
from soche.wav.constants import CHORDS_MINIMAL, NOTES_FREQ_MINIMAL
from soche.wav.sine import (
    generate_chord,
    generate_sin_array_normalized_with_shift,
    multi_array_sum,
)


def scale():
    n = []
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["C"], 0.2, 0.0))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["D"], 0.2, 0.2))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["E"], 0.2, 0.4))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["F"], 0.2, 0.6))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["G"], 0.2, 0.8))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["A"], 0.2, 1.0))
    n.append(generate_sin_array_normalized_with_shift(1, NOTES_FREQ_MINIMAL["B"], 0.2, 1.2))
    f = multi_array_sum(tuple(n))
    return f


def spanish():
    am = generate_chord(0.1, CHORDS_MINIMAL["Am"], 1, 0)
    g  = generate_chord(0.1, CHORDS_MINIMAL["G"], 1, 1)
    f  = generate_chord(0.1, CHORDS_MINIMAL["F"], 1, 2)
    e  = generate_chord(0.1, CHORDS_MINIMAL["E"], 1, 3)
    s = multi_array_sum((e, am, f, g))
    return s


def twinkle_twinkle_star():
    n = []
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["G"], 0.2, 0.0))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["G"], 0.2, 0.2))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["D"], 0.2, 0.4))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["D"], 0.2, 0.6))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["E"], 0.2, 0.8))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["E"], 0.2, 1.0))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["D"], 0.2, 1.2))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["C"], 0.2, 1.4))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["C"], 0.2, 1.6))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["B"], 0.2, 1.8))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["B"], 0.2, 2.0))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["A"], 0.2, 2.2))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["A"], 0.2, 2.4))
    n.append(generate_sin_array_normalized_with_shift(0.6, NOTES_FREQ_MINIMAL["G"], 0.2, 2.6))
    f = multi_array_sum(tuple(n))
    return f


# TODO: add more test signals
# TODO: add noise to test signals
