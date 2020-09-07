# -*- coding: utf-8 -*-
"""
Audio-to-MIDI converter using librosa


"""
import numpy as np
import librosa
import midiutil
import sys

def transition_matrix(note_min, note_max, p_stay_note, p_stay_silence):
    """
    Returns the transition matrix with one silence state and two states
    (onset and sustain) for each note.

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
    T : numpy 2x2 array
        Trasition matrix in which T[i,j] is the probability of
        going from state i to state j

    """
    
    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1
    p_ = (1-p_stay_silence)/n_notes
    p__ = (1-p_stay_note)/(n_notes+1)
    
    # Transition matrix:
    # State 0 = silence
    # States 1, 3, 5... = onsets
    # States 2, 4, 6... = sustains
    T = np.zeros((2*n_notes+1, 2*n_notes+1))

    # State 0: silence
    T[0,0] = p_stay_silence
    for i in range(n_notes):
        T[0, (i*2)+1] = p_
    
    # States 1, 3, 5... = onsets
    for i in range(n_notes):
        T[(i*2)+1, (i*2)+2] = 1

    # States 2, 4, 6... = sustains
    for i in range(n_notes):
        T[(i*2)+2, 0] = p__
        T[(i*2)+2, (i*2)+2] = p_stay_note
        for j in range(n_notes):        
            T[(i*2)+2, (j*2)+1] = p__
    
    return T


def probabilities(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc, onset_acc, spread):
    """
    Estimate prior (observed) probabilities from audio signal
    

    Parameters
    ----------
    y : 1-D numpy array
        Array containing audio samples
        
    note_min : string, 'A#4' format
        Lowest note supported by this estimator
    note_max : string, 'A#4' format
        Highest note supported by this estimator
    sr : int
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
    P : 2D numpy array.
        P[j,t] is the prior probability of being in state j at time t.

    """
    
    fmin = librosa.note_to_hz(note_min)
    fmax = librosa.note_to_hz(note_max)
    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1
    
    # F0 and voicing
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin*0.9, fmax*1.1, sr, frame_length, window_length, hop_length)
    tuning = librosa.pitch_tuning(f0)
    f0_ = np.round(librosa.hz_to_midi(f0-tuning)).astype(int)
    onsets = librosa.onset.onset_detect(y, sr=sr, hop_length=hop_length, backtrack=True)


    P = np.ones( (n_notes*2 + 1, len(f0)) )

    for t in range(len(f0)):
        # probability of silence or onset = 1-voiced_prob
        # Probability of a note = voiced_prob * (pitch_acc) (estimated note)
        # Probability of a note = voiced_prob * (1-pitch_acc) (estimated note)
        if voiced_flag[t]==False:
            P[0,t] = voiced_acc
        else:
            P[0,t] = 1-voiced_acc

        for j in range(n_notes):
            if t in onsets:
                P[(j*2)+1, t] = onset_acc
            else:
                P[(j*2)+1, t] = 1-onset_acc

            if j+midi_min == f0_[t]:
                P[(j*2)+2, t] = pitch_acc

            elif np.abs(j+midi_min-f0_[t])==1:
                P[(j*2)+2, t] = pitch_acc * spread

            else:
                P[(j*2)+2, t] = 1-pitch_acc

    return P

def states_to_pianoroll(states, note_min, note_max, hop_time):
    """
    Converts state sequence to an intermediate, internal piano-roll notation

    Parameters
    ----------
    states : int
        Sequence of states estimated by Viterbi
    note_min : string, 'A#4' format
        Lowest note supported by this estimator
    note_max : string, 'A#4' format
        Highest note supported by this estimator
    hop_time : float
        Time interval between two states.

    Returns
    -------
    output : List of lists
        output[i] is the i-th note in the sequence. Each note is a list
        described by [onset_time, offset_time, pitch].

    """
    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    
    states_ = np.hstack( (states, np.zeros(1)))
    
    # possible types of states
    silence = 0
    onset = 1
    sustain = 2

    my_state = silence
    output = []
    
    last_onset = 0
    last_offset = 0
    last_midi = 0
    for i in range(len(states_)):
        if my_state == silence:
            if int(states_[i]%2) != 0:
                # Found an onset!
                last_onset = i * hop_time
                last_midi = ((states_[i]-1)/2)+midi_min
                last_note = librosa.midi_to_note(last_midi)
                my_state = onset


        elif my_state == onset:
            if int(states_[i]%2) == 0:
                my_state = sustain

        elif my_state == sustain:
            if int(states_[i]%2) != 0:
                # Found an onset.                
                # Finish last note
                last_offset = i*hop_time
                my_note = [last_onset, last_offset, last_midi, last_note]
                output.append(my_note)
                
                # Start new note
                last_onset = i * hop_time
                last_midi = ((states_[i]-1)/2)+midi_min
                last_note = librosa.midi_to_note(last_midi)
                my_state = onset
            
            elif states_[i]==0:
                # Found silence. Finish last note.
                last_offset = i*hop_time
                my_note = [last_onset, last_offset, last_midi, last_note]
                output.append(my_note)
                my_state = silence

    return output


def pianoroll_to_midi(y, pianoroll):
    """
    

    Parameters
    ----------
    y : 1D numpy array.
        Audio signal (used to estimate BPM)
        
    pianoroll : list
        A pianoroll list as estimated by states_to_pianoroll().

    Returns
    -------
    None.

    """
    bpm = librosa.beat.tempo(y)[0]
    print(bpm)
    quarter_note = 60/bpm
    ticks_per_quarter = 1024
    
    onsets = np.array([p[0] for p in pianoroll])
    offsets = np.array([p[1] for p in pianoroll])
    
    onsets = onsets / quarter_note
    offsets = offsets  / quarter_note
    durations = offsets-onsets
    
    
    MyMIDI = midiutil.MIDIFile(1)
    MyMIDI.addTempo(0, 0, bpm)
    
    for i in range(len(onsets)):
        MyMIDI.addNote(0, 0, int(pianoroll[i][2]), onsets[i], durations[i], 100)

    return MyMIDI
        

def run(file_in, file_out):
    #sr=22050
    note_min='A2'
    note_max='E6'
    voiced_acc = 0.9
    onset_acc = 0.8
    frame_length=2048
    window_length=1024
    hop_length=256
    pitch_acc = 0.99
    spread = 0.6
    
    y, sr = librosa.load(file_in)

    T = transition_matrix(note_min, note_max, 0.9, 0.2)
    P = probabilities(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc, onset_acc, spread)
    p_init = np.zeros(T.shape[0])
    p_init[0] = 1
    
    states = librosa.sequence.viterbi(P, T, p_init=p_init)
    #print(states)
    pianoroll=states_to_pianoroll(states, note_min, note_max, hop_length/sr)
    #print(pianoroll)
    MyMIDI = pianoroll_to_midi(y, pianoroll)
    with open(file_out, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    

print("Welcome!")
file_in = sys.argv[1]
file_out = sys.argv[2]
print(sys.argv[1], sys.argv[2])    
run(file_in, file_out)

    
