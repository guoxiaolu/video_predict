import numpy as np
import os
from extractor import Extractor
from tqdm import tqdm
import glob


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the original list."""
    assert len(input_list) >= size

    samples = np.linspace(0, len(input_list)-1, size, dtype=int).tolist()
    output = [input_list[s] for s in samples]

    # Cut off the last one if needed.
    return output

# Set defaults.
seq_length = 200

video_path = 'data/video0'
frame_path = 'data/frame'
sequence_path = 'data/sequence'
if not os.path.exists(sequence_path):
    os.mkdir(sequence_path)

video_name = os.listdir(video_path)
video_name_noext = [name.split('.')[0] for name in video_name]

pbar = tqdm(total=len(video_name_noext))
model = Extractor()
for video in video_name_noext:
    img_list = glob.glob(os.path.join(frame_path, video+'_*.jpg'))
    if len(img_list) == 0:
        continue

    seqfile = os.path.join(sequence_path, video + '_' + str(seq_length) + '.npy')

    # Check if we already have it.
    if os.path.isfile(seqfile):
        pbar.update(1)
        continue

    img_list_sorted = sorted(img_list)
    frames = rescale_list(img_list_sorted, seq_length)

    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.save(seqfile, sequence)

    pbar.update(1)
pbar.close()
