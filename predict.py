import os
import numpy as np
from keras.models import load_model
import extract_frame
import extract_feature
import extract_audio
import extract_audio_feature

seq_length_frame = 400
seq_length_audio = 200
video_name = 'data/video0/00009.mp4'
model = load_model('model/weights.00300.hdf5')
model.summary()

video_name_noext = video_name.split(os.path.sep)[-1].split('.')[0]
save_path = './tmp/' + video_name_noext
if not os.path.exists(save_path):
    os.makedirs(save_path)

_, nb_frames = extract_frame.extract_one_file(video_name, save_path)
extract_feature.extract_one_feature(video_name_noext, save_path, save_path)
seqfile = os.path.join(save_path, video_name_noext + '_' + str(seq_length_frame) + '.npy')
sequence_frame = np.load(seqfile)

extract_audio.extract_one_file(video_name, save_path)
extract_audio_feature.extract_one_feature(video_name_noext, save_path, save_path)
seqfile = os.path.join(save_path, video_name_noext + '_audio_' + str(seq_length_audio) + '.npy')
sequence_audio = np.load(seqfile)

result = model.predict([np.expand_dims(sequence_frame, axis=0), np.expand_dims(sequence_audio, axis=0)])
print result