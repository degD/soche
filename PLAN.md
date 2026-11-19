
# Project Roadmap

## Phase 0: Setup and Input
- [x] Set a repo.
- [x] Read and write `.wav` files.
- [x] Generate syntethic signals.

## Phase 1: STFT and Chroma Features
- [x] Implement STFT with Hanning.
- [x] Convert frequency to MIDI note number and to class (0-11).
- [x] Implement harmonic separation with median filtering.
- [ ] Implement tuning estimation.

## Phase 2: Chord Templates
- [x] Define a "no chord", 12 major, 12 minor chord templates.
- [x] Classify frame chords using cosine similarity.
- [x] Implement the Viterbi algorithm.

## Phase 3: Pilot Dataset
- [ ] Define a lab result format.
- [ ] Write evaluation tool.
- [ ] Test chords of a 10 song syntethic dataset.
- [ ] Show confusion matrix.

## Phase 4: Existing Dataset
- [ ] Access a dataset with real songs and annotated chords.
- [ ] Add dominant seventh chords.
- [ ] Evaluate and tune against the dataset.
- [ ] Limit annotation to defined chords only and evaluate again.
- [ ] Group by errors and report.

## Phase 5: Modularize

- [ ] Turn project modular.
- [ ] Refactor the code.
- [ ] Turn into a CLI tool.

## Phase 6: Extensions
- [ ] Research spectogram improvements and document.
- [ ] Add beat-synchronous chroma.
- [ ] Add key estimation.
- [ ] Try a ML models.
- [ ] Try training a neaural network
- [ ] Try transformers.
