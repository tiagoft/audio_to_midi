# Sound to midi converter

Tools to convert audio files to midi. The CLI does not require any programming. The library enables stopping the process at any intermediate step.

Currently, supports monophonic audio (tunes played one note at a time).

Follow this guide for installation and usage.

For information about how it works, read the [whitepaper](monophonic_audio_to_midi.md).

## Quickstart guides

### Installation from pip

`pip install sound_to_midi`

### Installation from Github repo

`git clone https://github.com/tiagoft/audio_to_midi.git`

`cd audio_to_midi`

`python -m build`

`pip install .\dist\audio_to_midi-0.0.1-py3-none-any.whl`

### Usage as a Python library


    import sys
    import librosa

    from audio_to_midi.monophonic import wave_to_midi

    print("Starting...")
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    y, sr = librosa.load(file_in, sr=None)
    print("Audio file loaded!")
    midi = wave_to_midi(y, sr=sr)
    print("Conversion finished!")
    with open (file_out, 'wb') as f:
        midi.writeFile(f)
    print("Done. Exiting!")


### Command-line interface (CLI)

After installing:

`w2m input_file.wav output_file.mid`

(supports most common formats like wav, aiff, mp3...)





