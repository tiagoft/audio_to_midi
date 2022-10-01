# -*- coding: utf-8 -*-

import numpy as np
import librosa
import midiutil


def transition_matrix(
        note_min: str,
        note_max: str,
        p_stay_note: float,
        p_stay_silence: float) -> np.array:
    """
    Returns the transition matrix with one silence state and two states
    (onset and sustain) for each note. This matrix mixes an acoustic model with two states with
    an uniform-transition linguistic model.

    Parameters
    ----------
    note_min : string, 'A#4' format
        Lowest note supported by this transition matrix
    note_max : string, 'A#4' format
        Highest note supported by this transition matrix
    p_stay_note : float, between 0 and 1
        Probability of a sustain state returning to itself.
    p_stay_silence : float, between 0 and 1
        Probability of the silence state returning to itselt.

    Returns
    -------
    transmat : np.array (2*N_notes+1x2*N_notes+1)
        Trasition matrix in which t[i,j] is the probability of
        going from state i to state j

    """

    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1
    p_l = (1 - p_stay_silence) / n_notes
    p_ll = (1 - p_stay_note) / (n_notes + 1)

    # Transition matrix:
    # State 0 = silence
    # States 1, 3, 5... = onsets
    # States 2, 4, 6... = sustains
    transmat = np.zeros((2 * n_notes + 1, 2 * n_notes + 1))

    # State 0: silence
    transmat[0, 0] = p_stay_silence
    for i in range(n_notes):
        transmat[0, (i * 2) + 1] = p_l

    # States 1, 3, 5... = onsets
    for i in range(n_notes):
        transmat[(i * 2) + 1, (i * 2) + 2] = 1

    # States 2, 4, 6... = sustains
    for i in range(n_notes):
        transmat[(i * 2) + 2, 0] = p_ll
        transmat[(i * 2) + 2, (i * 2) + 2] = p_stay_note
        for j in range(n_notes):
            transmat[(i * 2) + 2, (j * 2) + 1] = p_ll

    return transmat


def prior_probabilities(
        audio_signal: np.array,
        note_min: str,
        note_max: str,
        srate: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        pitch_acc: float = 0.9,
        voiced_acc: float = 0.9,
        onset_acc: float = 0.9,
        spread: float = 0.2) -> np.array:
    """
    Estimate prior (observed) probabilities from audio signal

    Parameters
    ----------
    audio_signal : 1-D numpy array
        Array containing audio samples

    note_min : string, 'A#4' format
        Lowest note supported by this estimator
    note_max : string, 'A#4' format
        Highest note supported by this estimator
    srate : int
        Sample rate.
    frame_length : int
    window_length : int
    hop_length : int
        Parameters for FFT estimation
    pitch_acc : float, between 0 and 1
        Probability (estimated) that the pitch estimator is correct.
    voiced_acc : float, between 0 and 1
        Estimated accuracy of the "voiced" parameter.
    onset_acc : float, between 0 and 1
        Estimated accuracy of the onset detector.
    spread : float, between 0 and 1
        Probability that the singer/musician had a one-semitone deviation
        due to vibrato or glissando.

    Returns
    -------
    priors : 2D numpy array.
        priors[j,t] is the prior probability of being in state j at time t.

    """

    fmin = librosa.note_to_hz(note_min)
    fmax = librosa.note_to_hz(note_max)
    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1

    # pitch and voicing
    pitch, voiced_flag, _ = librosa.pyin(
        y=audio_signal, fmin=fmin * 0.9, fmax=fmax * 1.1,
        sr=srate, frame_length=frame_length, win_length=int(frame_length / 2),
        hop_length=hop_length)
    tuning = librosa.pitch_tuning(pitch)
    f0_ = np.round(librosa.hz_to_midi(pitch - tuning)).astype(int)
    onsets = librosa.onset.onset_detect(
        y=audio_signal, sr=srate,
        hop_length=hop_length, backtrack=True)

    priors = np.ones((n_notes * 2 + 1, len(pitch)))

    for n_frame in range(len(pitch)):
        # probability of silence or onset = 1-voiced_prob
        # Probability of a note = voiced_prob * (pitch_acc) (estimated note)
        # Probability of a note = voiced_prob * (1-pitch_acc) (estimated note)
        if not voiced_flag[n_frame]:
            priors[0, n_frame] = voiced_acc
        else:
            priors[0, n_frame] = 1 - voiced_acc

        for j in range(n_notes):
            if n_frame in onsets:
                priors[(j * 2) + 1, n_frame] = onset_acc
            else:
                priors[(j * 2) + 1, n_frame] = 1 - onset_acc

            if j + midi_min == f0_[n_frame]:
                priors[(j * 2) + 2, n_frame] = pitch_acc

            elif np.abs(j + midi_min - f0_[n_frame]) == 1:
                priors[(j * 2) + 2, n_frame] = pitch_acc * spread

            else:
                priors[(j * 2) + 2, n_frame] = 1 - pitch_acc

    return priors


def states_to_pianoroll(states: list, note_min: str, hop_time: float) -> list:
    """
    Converts state sequence to an intermediate, internal piano-roll notation

    Parameters
    ----------
    states : list of int (or other iterable)
        Sequence of states estimated by Viterbi
    note_min : string, 'A#4' format
        Lowest note supported by this estimator
    hop_time : float
        Time interval between two states.

    Returns
    -------
    output : List of lists
        output[i] is the i-th note in the sequence. Each note is a list
        described by [onset_time, offset_time, pitch, note_name], e.g., output[1][0]
        is the onset time for the second note.
    """
    midi_min = librosa.note_to_midi(note_min)

    states_ = np.hstack((states, np.zeros(1)))

    # possible types of states
    silence = 0
    onset = 1
    sustain = 2

    my_state = silence
    output = []

    last_onset = 0
    last_offset = 0
    last_midi = 0
    for i, _ in enumerate(states_):
        if my_state == silence:
            if int(states_[i] % 2) != 0:
                # Found an onset!
                last_onset = i * hop_time
                last_midi = ((states_[i] - 1) / 2) + midi_min
                last_note = librosa.midi_to_note(last_midi)
                my_state = onset

        elif my_state == onset:
            if int(states_[i] % 2) == 0:
                my_state = sustain

        elif my_state == sustain:
            if int(states_[i] % 2) != 0:
                # Found an onset.
                # Finish last note
                last_offset = i * hop_time
                my_note = [last_onset, last_offset, last_midi, last_note]
                output.append(my_note)

                # Start new note
                last_onset = i * hop_time
                last_midi = ((states_[i] - 1) / 2) + midi_min
                last_note = librosa.midi_to_note(last_midi)
                my_state = onset

            elif states_[i] == 0:
                # Found silence. Finish last note.
                last_offset = i * hop_time
                my_note = [last_onset, last_offset, last_midi, last_note]
                output.append(my_note)
                my_state = silence

    return output


def pianoroll_to_midi(bpm: float, pianoroll: list) -> midiutil.MIDIFile():
    """
    Converts an internal piano roll notation to a MIDI file

    Parameters
    ----------
    bpm: float
        Beats per minute for the MIDI file. If necessary, use
        bpm = librosa.beat.tempo(y)[0] to estimate bpm.

    pianoroll : list
        A pianoroll list as estimated by states_to_pianoroll().

    Returns
    -------
    None.

    """
    quarter_note = 60 / bpm

    onsets = np.array([p[0] for p in pianoroll])
    offsets = np.array([p[1] for p in pianoroll])

    onsets = onsets / quarter_note
    offsets = offsets / quarter_note
    durations = offsets - onsets

    midi = midiutil.MIDIFile(1)
    midi.addTempo(0, 0, bpm)

    for i, _ in enumerate(onsets):
        midi.addNote(
            0, 0, int(pianoroll[i][2]), onsets[i], durations[i], 100)

    return midi


def wave_to_midi(
        audio_signal: np.array,
        srate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        note_min: str = "A2",
        note_max: str = "E5",
        p_stay_note: float = 0.9,
        p_stay_silence: float = 0.7,
        pitch_acc: float = 0.9,
        voiced_acc: float = 0.9,
        onset_acc: float = 0.9,
        spread: float = 0.2) -> midiutil.MIDIFile():
    """Converts an audio signal to a MIDI file

    Args:
        audio_signal (np.array): Array containing audio samples
        srate (int, optional): Sample rate of the audio signal Defaults to 22050.
        frame_length (int, optional): Frame length for analysis. Defaults to 2048.
        hop_length (int, optional): Hop between two frames in analysis. Defaults to 512.
        note_min (str, optional): Lowest allowed note in "A#4" format. Defaults to "A2".
        note_max (str, optional): Highest allowed note in "A#4" format. Defaults to "E5".
        p_stay_note (float, optional): Probability of staying in the same note for two
                                                 subsequent frames. Defaults to 0.9.
        p_stay_silence (float, optional): Probability of staying in the silence state for
                                                 two subsequent frames. Defaults to 0.7.
        pitch_acc (float, optional): Probability (reliability) that the pitch estimator
                                                 is correct. Defaults to 0.9.
        voiced_acc (float, optional): Estimated accuracy of the "voiced" parameter.
                                                 Defaults to 0.9.
        onset_acc (float, optional): Estimated accuracy of the onset detector. Defaults to 0.9.
        spread (float, optional): Probability that the audio signal deviates by one semitone
                                                 due to vibrato or glissando. Defaults to 0.2.

    Returns:
        midi (midiutil.MIDIFile): A MIDI file that can be written to disk.
    """
    transmat = transition_matrix(note_min, note_max, p_stay_note, p_stay_silence)
    priors = prior_probabilities(
        audio_signal,
        note_min,
        note_max,
        srate,
        frame_length,
        hop_length,
        pitch_acc,
        voiced_acc,
        onset_acc,
        spread)
    p_init = np.zeros(transmat.shape[0])
    p_init[0] = 1
    states = librosa.sequence.viterbi(priors, transmat, p_init=p_init)

    pianoroll = states_to_pianoroll(states, note_min, hop_length / srate)
    bpm = librosa.beat.tempo(y=audio_signal)[0]
    midi = pianoroll_to_midi(bpm, pianoroll)

    return midi
