
# Project Roadmap

## Phase 0: Project setup
- [x] Set a venv with numpy, matplotlib, and pytest.
- [x] Write a README that states the goal, chord vocabulary, and sample rate.

## Phase 1: Audio input
- [x] Read mono WAV files with the `wave` module and convert them to float numpy arrays in the range −1 to 1.
- [x] Write a tool for writing WAV files, and to create test signals.
- [x] Test if signal write and read preserves its values.

## Phase 2: Synthetic test signals
- [x] Write a generator that produces chords from sums of sine waves at equal-tempered frequencies, with A4 = 440 Hz.
- [x] Generate a set of known progressions (for example, C–Am–F–G).
- [ ] Add noise and amplitude changes to check robustness.

## Phase 3: Short-time Fourier transform
- [x] Implement framing with a hop of 512 samples and a window of 4096 samples, using a strided view.
- [x] Apply a Hann window and compute the real FFT for each frame.
- [x] Produce a magnitude spectrogram and plot it with matplotlib.
- [x] Verify that a pure sine at a known frequency peaks at the expected bin.

## Phase 4: Chroma features
- [x] Convert frequency to MIDI note number, and MIDI note number to pitch class (0 to 11).
- [x] Build a filterbank that maps FFT bins to pitch classes, with each frequency contributing to its nearest semitone.
- [x] Compute the chroma matrix (12 × frames) by multiplying the spectrogram with the filterbank.
- [x] Normalize each chroma frame, and check that a synthetic C major triad peaks at C, E, and G.

## Phase 5: Pre-processing
- [ ] Implement tuning estimation: histogram the fractional semitone deviations of spectral peaks, then shift the filterbank by the estimated offset.
- [x] Implement harmonic/percussive separation with median filtering along time and frequency on the spectrogram, then keep the harmonic part.
- [x] Compare chroma with and without each step on a few real recordings, and keep notes on the effect.

## Phase 6: Chord templates and frame-wise decoding
- [x] Define 24 major and minor triad templates, plus a "no chord" class (25 classes in total).
- [x] Score each frame against every template using cosine similarity, and pick the best match.
- [ ] Add a rule for the no-chord class, based on a threshold on the best score.
- [x] Build this frame-wise baseline and check that it works on synthetic data before using real songs.

## Phase 7: Viterbi decoding
- [ ] Implement the Viterbi algorithm in log-space with backpointers and backtracking.
- [ ] Define the transition matrix with a tunable self-transition probability.
- [ ] Test on synthetic progressions and confirm that smoothing removes single-frame flickers.
- [ ] Compare frame-wise and Viterbi outputs on the same songs.

## Phase 8: Annotation format and data handling
- [ ] Choose the lab format (start time, end time, chord label per line) and write a parser.
- [ ] Write a function that converts segment annotations to per-frame labels using the hop size.
- [ ] Write validation checks: no overlapping segments, labels from the vocabulary, times in order.

## Phase 9: Evaluation
- [ ] Implement frame-level accuracy.
- [ ] Implement root accuracy and major/minor accuracy, where the no-chord class is handled explicitly.
- [ ] Produce a confusion matrix and a per-song report.
- [ ] Add a plot that overlays the reference and predicted chord sequences over time.

## Phase 10: Pilot dataset
- [ ] Choose 10 to 20 pop songs and confirm your rights to use them for research.
- [ ] Annotate them manually, using Sonic Visualiser or a similar tool, and export in the lab format.
- [ ] Annotate a few songs twice, at different times, to measure your own consistency.
- [ ] Run the full pipeline on the pilot set and record baseline scores.

## Phase 11: Error analysis and tuning
- [ ] Group errors by type: relative-key confusions (C major and A minor), wrong root, inversions, and no-chord mistakes.
- [ ] Tune the self-transition probability, template weights, and the no-chord threshold on a separate tuning split.
- [ ] Decide whether dominant 7th chords are worth adding, which would grow the vocabulary to 73 classes (24 major, 24 minor, 24 dominant 7th, plus no chord). Use pop data to justify the choice.

## Phase 12: Baseline evaluation
- [ ] Expand the annotated set to 50 to 100 songs, with a fixed train/tune/test split.
- [ ] Freeze the pipeline and report results on the test split once, without further tuning.
- [ ] Add regression tests that fail if scores drop on a small fixed benchmark.

## Phase 13: Command-line tool
- [ ] Build a command-line interface with `argparse` that takes a WAV file and writes a lab-format chord file.
- [ ] Add options for the sample rate, the decoder, and the output path.
- [ ] Write documentation for installation and usage, and test the tool end to end on a new song.

## Phase 14: Extensions (optional)
- [ ] Add beat-synchronous chroma, so chord changes align with the bars.
- [ ] Add key estimation, which can help disambiguate relative keys.
- [ ] Expand the dataset toward 300 or more songs.
- [ ] Try a small learned model, such as a neural network.
