from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from custom_generator import frame_generator

nb_epoch = 500
seq_length = 200
batch_size = 32
sequence_path = 'data/sequence'
train_file = 'data/train.txt'
test_file = 'data/test.txt'
tf = open(train_file, 'r')
lines = tf.readlines()
y_train = []
for line in lines:
    con = line.strip().split('\t')
    y_train.append([con[0], float(con[1]), float(con[2])])

tf = open(test_file, 'r')
lines = tf.readlines()
y_test = []
for line in lines:
    con = line.strip().split('\t')
    y_test.append([con[0], float(con[1]), float(con[2])])
tf.close()

train_generator = frame_generator(sequence_path, seq_length, y_train, batch_size)
test_generator = frame_generator(sequence_path, seq_length, y_test, 1)

input = Input(shape=(seq_length, 2048,), name='input')
x = LSTM(2048, return_sequences=False, dropout=0.75, name='lstm1')(input)
x = Dense(1024, activation='relu', name='dense1')(x)
x = Dropout(0.75, name='dropout_1')(x)
out = Dense(1, activation='sigmoid', name='out')(x)

model = Model(inputs=input, outputs=out)
model.summary()

sgd = SGD(lr=0.001, momentum=0.9, decay=1e-3, nesterov=False)
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])

tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./model/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
model.fit_generator(train_generator, len(y_train)/batch_size, nb_epoch=nb_epoch, callbacks=[tb, mc], validation_data=test_generator, validation_steps=len(y_test), initial_epoch=0)
