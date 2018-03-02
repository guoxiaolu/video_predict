from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from custom_generator import frame_generator

nb_epoch = 500
seq_length_frame = 400
seq_length_audio = 200
feature_length_frame = 2048
feature_length_audio = 128
batch_size = 32
sequence_path_frame = 'data/sequence'
sequence_path_audio = 'data/audio_sequence'
train_file = 'data/train.txt'
test_file = 'data/test.txt'
tf = open(train_file, 'r')
lines = tf.readlines()
y_train = []
for line in lines:
    con = line.strip().split('\t')
    y_train.append([con[0], float(con[1]), float(con[2]), float(con[3])])

tf = open(test_file, 'r')
lines = tf.readlines()
y_test = []
for line in lines:
    con = line.strip().split('\t')
    y_test.append([con[0], float(con[1]), float(con[2]), float(con[3])])
tf.close()

train_generator_frame = frame_generator(sequence_path_frame, seq_length_frame, y_train, batch_size)
test_generator_frame = frame_generator(sequence_path_frame, seq_length_frame, y_test, 1)

train_generator_audio = frame_generator(sequence_path_audio, seq_length_audio, y_train, batch_size)
test_generator_audio = frame_generator(sequence_path_audio, seq_length_audio, y_test, 1)

input_frame = Input(shape=(seq_length_frame, feature_length_frame,), name='input_frame')
x_frame = BatchNormalization()(input_frame)
x_frame = Bidirectional(LSTM(512, return_sequences=False, dropout=0.25, name='lstm_frame'))(x_frame)
x_frame = Dense(128, activation='relu', name='dense_frame')(x_frame)

input_audio = Input(shape=(seq_length_audio, feature_length_audio,), name='input_audio')
x_audio = BatchNormalization()(input_audio)
x_audio = Bidirectional(LSTM(32, return_sequences=False, dropout=0.25, name='lstm_audio'))(x_audio)
x_audio = Dense(8, activation='relu', name='dense_audio')(x_audio)

concat = Concatenate(name='concate')([x_frame, x_audio])
out = Dense(1, activation='sigmoid', name='out')(concat)

model = Model(inputs=[input_frame, input_audio], outputs=out)
model.summary()

sgd = SGD(lr=0.001, momentum=0.9, decay=1e-3, nesterov=False)
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])

tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./model/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False,
                     save_weights_only=False, mode='auto', period=10)
model.fit_generator([train_generator_frame,train_generator_audio], len(y_train)/batch_size, nb_epoch=nb_epoch,
                    callbacks=[tb, mc], validation_data=[test_generator_frame,test_generator_audio],
                    validation_steps=len(y_test), initial_epoch=0)
