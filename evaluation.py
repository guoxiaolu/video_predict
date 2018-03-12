import os
import numpy as np
from keras.models import load_model

seq_length_frame = 400
seq_length_audio = 200
seq_path_frame = './data/sequence'
seq_path_audio = './data/audio_sequence'
test_flie = './data/test.txt'
model = load_model('./model/weights.00500.hdf5')
model.summary()

f = open(test_flie, mode='r')
lines = f.readlines()
f.close()

for line in lines:
    con = line.strip().split('\t')
    name = con[0]
    gt = con[3]
    frame_path = os.path.join(seq_path_frame, name.split('.')[0] + '_' + str(seq_length_frame) + '.npy')
    audio_path = os.path.join(seq_path_audio, name.split('.')[0] + '_' + str(seq_length_audio) + '.npy')
    sequence_frame = np.load(frame_path)
    sequence_audio = np.load(audio_path)
    result = model.predict([np.expand_dims(sequence_frame, axis=0), np.expand_dims(sequence_audio, axis=0)])[0][0]
    print '%s\t%s\t%f'%(result, gt, result)
