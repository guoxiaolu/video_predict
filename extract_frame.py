import csv
import glob
import os
import os.path
from subprocess import call

def extract_files(folder='data/video', dst_path='data/frame'):
    """After we have all of our videos, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos.

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    data_files = glob.glob(os.path.join(folder, '*.mp4'))

    for video_path in data_files:
        filename_no_ext, nb_frames = extract_one_file(video_path, dst_path)
        data_file.append([filename_no_ext, nb_frames])

    with open('data/data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def extract_one_file(video_path, dst_path):
    # Get the parts of the file.
    filename_no_ext, filename = get_video_parts(video_path)

    # Only extract if we haven't done it yet. Otherwise, just get
    # the info.
    if not check_already_extracted(filename_no_ext, dst_path):
        # Now extract it.
        dest = os.path.join(dst_path, filename_no_ext + '_%05d.jpg')
        call(["ffmpeg", "-i", video_path, dest])

    # Now get how many frames it is.
    nb_frames = get_nb_frames_for_video(filename_no_ext, dst_path)
    print("Generated %d frames for %s" % (nb_frames, filename_no_ext))
    return filename_no_ext, nb_frames

def get_nb_frames_for_video(filename_no_ext, dst_path):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    generated_files = glob.glob(os.path.join(dst_path,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]

    return filename_no_ext, filename

def check_already_extracted(filename_no_ext, dst_path):
    """Check to see if we created the -0001 frame of this file."""
    return bool(os.path.exists(os.path.join(dst_path,
                               filename_no_ext + '_%05d.jpg'%(1))))

def main():
    extract_files()

if __name__ == '__main__':
    main()
