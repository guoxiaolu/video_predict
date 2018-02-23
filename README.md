video click predict based on temporal contents, this is a regression question of video based on cnn+rnn method.

Data collection:
Get car ad video from iqiyi and get related information such as title, play number, like number. Then process the data just as the excel shown. Finally, remove most video with almost zero ground truth as the data unbalancement. This data preprocess is mainly caused by the small number of video and low-dimension ground-truth.

Data preparation:
1. extract frames of video.
2. extract frame features(inception-v3 or resnet50, get the global_avg output, 2048-dimension) and conactenate theses features to a fixed sequence.

Training:
Train the sequence based on rnn method.
1. lstm units number seems have no big relation with final loss.
2. gru is not better than lstm.
3. 2-layers lstm is not better than 1-layer lstm.
4. bidirection lstm is better than native lstm.
5. inception-v3 seems better, even than ensemble feature of inception-v3 and resnet50.
Currently get the validation loss with about 0.23.

Prediction:
input is one video.

Currently, audio features can be added just as youtube8m method. Related codes come from: https://github.com/tensorflow/models/tree/master/research/audioset

More improvment...
