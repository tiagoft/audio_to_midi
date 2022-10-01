# Audio to midi converter

Follow this guide for installation and usage.

For information on how it works, check the [whitepaper](monophonic_audio_to_midi.md).

## Quickstart guides

### Installation
`pip3 install numpy librosa midiutil`

`git clone https://github.com/tiagoft/audio_to_midi.git`

### Using as python library

The source code for `cli.py` can be an example:

    import sys
    import librosa

    from audio2midi import wave_to_midi

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


### Using as command line tool

`cd audio_to_midi`

`python3 src/cli.py input_wav_file.wav output_midi_file.mid`





