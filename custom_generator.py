import numpy as np
import os
from random import shuffle

def frame_generator(sequence_path, seq_length, y_list, batch_size=32):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:
    data_type: 'features', 'images'
    """
    num = len(y_list)
    while True:
        start = 0
        # shuffle the list after one epoch
        shuffle(y_list)
        while True:
            X, y = [], []
            if start >= num:
                start = 0

            # Generate batch_size samples.
            for i, v in enumerate(y_list):
                if i < start:
                    continue
                # Reset to be safe.
                name = v[0]
                value = v[-1:]
                sequence = None

                path = os.path.join(sequence_path, name.split('.')[0]+'_'+str(seq_length)+'.npy')
                sequence = np.load(path)

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X.append(np.array(sequence))
                y.append(np.array(value))
                if len(X) == batch_size or i == num - 1:
                    start = i+1
                    break

            yield np.array(X), np.array(y)

def combined_generator(sequence_path_frame, seq_length_frame, sequence_path_audio, seq_length_audio, y_list, batch_size=32):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:
    data_type: 'features', 'images'
    """
    num = len(y_list)
    while True:
        start = 0
        # shuffle the list after one epoch
        shuffle(y_list)
        while True:
            X_frame, X_audio, y = [], [], []
            if start >= num:
                start = 0

            # Generate batch_size samples.
            for i, v in enumerate(y_list):
                if i < start:
                    continue
                # Reset to be safe.
                name = v[0]
                value = v[-1:]

                sequence_frame = None
                path = os.path.join(sequence_path_frame, name.split('.')[0] + '_' + str(seq_length_frame) + '.npy')
                sequence_frame = np.load(path)
                if sequence_frame is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                sequence_audio = None
                path = os.path.join(sequence_path_audio, name.split('.')[0] + '_' + str(seq_length_audio) + '.npy')
                sequence_audio = np.load(path)

                if sequence_audio is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X_frame.append(np.array(sequence_frame))
                X_audio.append(np.array(sequence_audio))
                y.append(np.array(value))
                if len(X_frame) == batch_size or i == num - 1:
                    start = i + 1
                    break

            yield [np.array(X_frame), np.array(X_audio)], np.array(y)
