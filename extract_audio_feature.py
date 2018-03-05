import numpy as np
import os
from tqdm import tqdm
import glob
from audio.vggish_inference import audio_inference

def extract_feature(video_path='data/video', audio_path='data/audio', sequence_path='data/audio_sequence',
                    seq_length=200, feature_length=128):
    if not os.path.exists(sequence_path):
        os.mkdir(sequence_path)

    video_name = glob.glob(os.path.join(video_path, '*.mp4'))
    video_name_noext = [name.split(os.path.sep)[-1].split('.')[0] for name in video_name]

    pbar = tqdm(total=len(video_name_noext))

    for video in video_name_noext:
        wav_file = os.path.join(audio_path, video+'.wav')
        if not os.path.isfile(wav_file):
            continue

        seqfile = os.path.join(sequence_path, video + '_' + str(seq_length) + '.npy')
        # Check if we already have it.
        if os.path.isfile(seqfile):
            pbar.update(1)
            continue

        sequence = audio_inference(wav_file)
        sequence_length = sequence.shape[0]
        if sequence_length >= seq_length:
            sequence = sequence[:seq_length,]
        else:
            tmp = np.zeros((seq_length, feature_length), dtype=np.uint8)
            tmp[:sequence_length,] = sequence
            sequence = tmp
        # Save the sequence.
        np.save(seqfile, sequence)

        pbar.update(1)
    pbar.close()

def extract_one_feature(video, audio_path, sequence_path, seq_length=200, feature_length=128):
    wav_file = os.path.join(audio_path, video + '.wav')
    if not os.path.isfile(wav_file):
        return

    seqfile = os.path.join(sequence_path, video + '_audio_' + str(seq_length) + '.npy')
    # Check if we already have it.
    if os.path.isfile(seqfile):
        return

    sequence = audio_inference(wav_file)
    sequence_length = sequence.shape[0]
    if sequence_length >= seq_length:
        sequence = sequence[:seq_length, ]
    else:
        tmp = np.zeros((seq_length, feature_length), dtype=np.uint8)
        tmp[:sequence_length, ] = sequence
        sequence = tmp
    # Save the sequence.
    np.save(seqfile, sequence)

def main():
    extract_feature()

if __name__ == '__main__':
    main()
