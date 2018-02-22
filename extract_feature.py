import numpy as np
import os
from extractor import Extractor
from tqdm import tqdm
import glob


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the original list.
    If the list size is smaller than size, will append -1 to make the return
    list size is same as 'size'."""
    # assert len(input_list) >= size
    list_size = len(input_list)
    if list_size >= size:
        samples = np.linspace(0, len(input_list)-1, size, dtype=int).tolist()
        output = [input_list[s] for s in samples]
    else:
        output = input_list + [-1] * (size - list_size)

    return output

def extract_feature(video_path='data/video', frame_path='data/frame', sequence_path='data/sequence',
                    seq_length=400, feature_length=4096):
    if not os.path.exists(sequence_path):
        os.mkdir(sequence_path)

    video_name = glob.glob(os.path.join(video_path, '*.mp4'))
    video_name_noext = [name.split(os.path.sep)[-1].split('.')[0] for name in video_name]

    pbar = tqdm(total=len(video_name_noext))
    model = Extractor()
    model_resnet50 = Extractor(model_name='resnet50')
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
            if image != -1:
                features = model.extract(image)
                features_resnet50 = model_resnet50.extract(image)
                features = np.concatenate((features, features_resnet50.flatten()))
            else:
                # zero paddind to the end of the list
                features = np.zeros((feature_length, ),dtype='float32')
            sequence.append(features)

        # Save the sequence.
        np.save(seqfile, sequence)

        pbar.update(1)
    pbar.close()

def extract_one_feature(video, frame_path, sequence_path, seq_length=400, feature_length=4096):
    model = Extractor()
    model_resnet50 = Extractor(model_name='resnet50')
    img_list = glob.glob(os.path.join(frame_path, video + '_*.jpg'))
    if len(img_list) == 0:
        return

    seqfile = os.path.join(sequence_path, video + '_' + str(seq_length) + '.npy')
    # Check if we already have it.
    if os.path.isfile(seqfile):
        return

    img_list_sorted = sorted(img_list)
    frames = rescale_list(img_list_sorted, seq_length)

    sequence = []
    for image in frames:
        if image != -1:
            features = model.extract(image)
            features_resnet50 = model_resnet50.extract(image)
            features = np.concatenate((features, features_resnet50.flatten()))
        else:
            features = np.zeros((feature_length, 1))
        sequence.append(features)

    # Save the sequence.
    np.save(seqfile, sequence)

def main():
    extract_feature()

if __name__ == '__main__':
    main()
