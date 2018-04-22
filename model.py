from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras import regularizers
from data_reader import Data
from matplotlib import pyplot as plt


def main():
    # the batch shape is [batch_size, 160, 320, 3]
    row, col, ch = 66, 200, 3  # Trimmed image format
    epoch = 25
    batch_size = 128
    data = Data(batch_size=batch_size)


    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', data_format='channels_last',
                     activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', data_format='channels_last',
                     activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', data_format='channels_last',
                     activation='relu'))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', data_format='channels_last',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', data_format='channels_last',
                     activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(data.train_generator, samples_per_epoch=len(data.train)/batch_size,
                        validation_data=data.validation_generator, validation_steps=len(data.val)/batch_size, nb_epoch=epoch)

    model.save('model1.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
