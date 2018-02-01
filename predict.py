import os
import numpy as np
from keras.models import load_model
from extract_frame import extract_one_file
from extract_feature import extract_one_feature

seq_length = 200
video_name = 'data/video0/00001.mp4'
model = load_model('model/weights.00200.hdf5')

video_name_noext = video_name.split(os.path.sep)[-1].split('.')[0]
save_path = './tmp/' + video_name_noext
if not os.path.exists(save_path):
    os.makedirs(save_path)

_, nb_frames = extract_one_file(video_name, save_path)
extract_one_feature(video_name_noext, save_path, save_path)
seqfile = os.path.join(save_path, video_name_noext + '_' + str(seq_length) + '.npy')
sequence = np.load(seqfile)
result = model.predict(np.expand_dims(sequence, axis=0))
print result