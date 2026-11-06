
import numpy as np

from soche.read_wave import read_wave
from soche.write_wave import (
    de_normalize_float_array,
    generate_sin_array_normalized,
    write_wave,
)

if __name__ == "__main__":
    p = generate_sin_array_normalized(5, 100, 1.1)
    write_wave("audio/compare_test.wav", p)
    q = read_wave("audio/compare_test.wav")

    p = de_normalize_float_array(p)
    q = de_normalize_float_array(q)

    if np.array_equal(p, q):
        print("SUCCESS: Read equals write.")
    else:
        print("FAILURE: Find what's wrong.")
