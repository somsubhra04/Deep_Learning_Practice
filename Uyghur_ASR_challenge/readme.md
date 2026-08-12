Your mission is to build a high-performing ASR system capable of accurately transcribing audio clips in the Uyghur language.

This competition provides you with a substantial dataset of over 23 hours of Uyghur speech. You will use this data to train a model that can listen to an audio file and output the corresponding text transcription.

Freedom to Innovate

Unlike the previous challenge, this competition has no restrictions on models, compute, or external data. You are encouraged to explore the full landscape of ASR techniques. Whether you choose to fine-tune massive pre-trained models like Whisper, leverage self-supervised methods like Wav2Vec2, or design a novel architecture from scratch, the choice is yours. You are free to use any tools, libraries, and computational resources at your disposal.

The goal is to push the boundaries of ASR performance for this low-resource language. We look forward to seeing the creative and powerful solutions you develop!

## Dataset Description
The dataset for this competition is comprised of audio files and corresponding metadata CSVs. Here’s what you'll find in the input directory:

wavs/: This folder contains all 9,468 audio clips in .wav format. Each file has a unique UUID as its name. All audio is single-channel (mono) and has a sample rate of 16,000 Hz. The total duration of the audio is approximately 23.95 hours.

train.csv: This file contains the metadata for the 7,574 samples in the training set. It has three columns:

ID: A unique identifier for the audio clip.

filepath: The relative path to the corresponding audio file within the wavs/ folder.

transcription: The ground-truth transcription for the audio file.

test.csv: This file contains the filepaths for the 1,894 samples in the test set. You will generate predictions for these files. It has two columns: ID and filepath.

sample.csv: A sample submission file in the correct format. It contains the IDs from the test set and example transcriptions.

## Metric: Character Error Rate (CER)

Your submissions will be evaluated using the Character Error Rate (CER), the standard metric for ASR tasks. CER measures the number of character-level errors in your predicted transcription compared to the ground truth. It is calculated as the Levenshtein distance—the minimum number of single-character edits (insertions, deletions, and substitutions) required to change the prediction to the ground truth—normalized by the total number of characters in the ground truth.

A lower CER score is better, with a score of 0 representing a perfect transcription.

Submission File Format

Your submission must be a CSV file named submission.csv with exactly two columns: ID and transcription. The ID for each row must match an ID from the provided test.csv file.
