import os
import random
import numpy as np
from PIL.Image import ANTIALIAS
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam

class Model:

    def __init__(self, training_data, model_path):
        self.training_data = training_data
        self.input_shape = self.training_data[0].shape[1:]
        self.classes = self.training_data[1].shape[1]
        self.checkpoint_cb = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
        self.X = self.training_data[0]
        self.Y = self.training_data[1]
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), data_format="channels_last", input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3),  data_format="channels_last"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3),  data_format="channels_last"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3),  data_format="channels_last"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.classes))
        model.add(Activation('softmax'))

        optimizer = Adam(lr=0.00001, decay=8e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def train(self, shuffle=True):

        if shuffle:
            idxes = [x for x in range(len(self.X))]
            random.shuffle(idxes)
            X_NEW = []
            Y_NEW = []

            for i in idxes:
                X_NEW.append(self.X[i])
                Y_NEW.append(self.Y[i])

            X = np.array(X_NEW)
            Y = np.array(Y_NEW)

        acc = self.model.fit(
            X,
            Y,
            batch_size=8,
            epochs=300,
            verbose=1,
            callbacks=[self.checkpoint_cb],
            validation_split=0.4
        )

        return acc

    def predict(self, X):
        answer = self.model.predict(np.array([X]))
        idx = np.argmax(answer)
        return self.classes[idx]
