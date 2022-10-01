# -*- coding: utf-8 -*-

import sys

import librosa

from audio2midi import wave_to_midi


if __name__ == "__main__":
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    y, sr = librosa.load(file_in, sr=None)
    midi = wave_to_midi(y, sr=sr)
    with open (file_out, 'wb') as f:
        f.write(midi)
    



