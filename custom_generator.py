import numpy as np
import os

def frame_generator(sequence_path, seq_length, y_list, batch_size=32):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:
    data_type: 'features', 'images'
    """
    num = len(y_list)
    while True:
        start = 0
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
                value = v[1:]
                sequence = None

                path = os.path.join(sequence_path, name.split('.')[0]+'_'+str(seq_length)+'.npy')
                sequence = np.load(path)

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(value)
                if len(X) == batch_size or i == num - 1:
                    start = i+1
                    break

            yield np.array(X), np.array(y)
