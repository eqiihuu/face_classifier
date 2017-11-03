import csv
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam


class ConvolutionNNs:
    TRAIN_DATA = "/Users/SiriusR/Documents/EE576/fer2013/train.csv"
    TEST_DATA = "/Users/SiriusR/Documents/EE576/fer2013/test.csv"

    def __init__(self):
        pass

    def neural_networks(self):
        batch_size = 32
        num_classes = 7
        epochs = 200
        train_x, train_y = self.read_data(self.TRAIN_DATA)
        test_x, test_y = self.read_data(self.TEST_DATA)
        train_x /= 256
        test_x /= 256
        train_y = np_utils.to_categorical(train_y)
        test_y = np_utils.to_categorical(test_y)
        model = Sequential()

        model.add(Conv2D(32, 3, 3, input_shape=(48, 48, 1)))

        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(train_x, train_y, validation_data=(test_x, test_y), nb_epoch=epochs, batch_size=batch_size)

        scores = model.evaluate(test_x, test_y, verbose=0)
        print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))
        model.save('/Users/SiriusR/Documents/EE576/ComputerVision/cnn.h5')

    def read_data(self, data_path):
        with open(data_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            rows = [row[:] for row in reader]
        x = rows[1:]
        # y = rows[0]
        # y = y[1:]
        x = np.asarray(x, dtype=float)
        y = x[:, 0]
        x = np.delete(x, 0, 1)
        new_x = []
        for row in x:
            new_x.append(row.reshape(48, 48))
        new_x = np.asarray(new_x)
        l = new_x.shape[0]
        x = new_x.reshape(l, 48, 48, 1)
        y = np.asarray(y, dtype=int)
        return x, y


if __name__ == '__main__':
    cnn = ConvolutionNNs()
    cnn.neural_networks()
