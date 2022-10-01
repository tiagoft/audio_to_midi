# -*- coding: utf-8 -*-

import numpy as np
import librosa
import midiutil
import sys

def transition_matrix(note_min : str, note_max: str, p_stay_note: float, p_stay_silence: float) -> np.array( 2*N_notes+1, 2*N_notes+1):
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
    T : np.array (2*N_notes+1x2*N_notes+1)
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


def states_to_pianoroll(states : list(int), note_min : str, note_max : str, hop_time : float) -> list:
    """
    Converts state sequence to an intermediate, internal piano-roll notation

    Parameters
    ----------
    states : list of int (or other iterable)
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
        described by [onset_time, offset_time, pitch, note_name], e.g., output[1][0]
        is the onset time for the second note.
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