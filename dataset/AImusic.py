import os
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream



# Step 1: Prepare the dataset and define network_input
def get_notes():
    notes = []
    for file in os.listdir('.'):
        if file.endswith(".mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

notes = get_notes()

# Construct the network input sequences
sequence_length = 100
n_vocab = len(set(notes))
network_input = []
network_output = []

# Create input sequences and the corresponding output
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

# Reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# Step 2: Build the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_vocab, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Step 3: Train the model
model.fit(network_input, network_output, epochs=200, batch_size=64)

# Save the trained model
model.save('music_generation_model.h5')

# Step 4: Generate music
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

# Generate 500 notes/chords
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    
    # Append the index to the pattern and then reshape
    pattern = np.append(pattern, index)
    pattern = pattern[1:]  # Remove the first element
    pattern = np.reshape(pattern, (len(pattern), 1))  # Reshape


# Convert the generated notes into a music21 stream
output_notes = []
for pattern in prediction_output:
    # If the pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        output_notes.append(new_chord)
    # If the pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

# Create a music21 stream object and add the generated notes
midi_stream = stream.Stream(output_notes)

# Save the generated music to a MIDI file
midi_stream.write('midi', fp='generated_music.mid')
