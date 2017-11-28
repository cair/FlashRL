import random

import numpy as np
from PIL.Image import ANTIALIAS
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
import os

from keras.optimizers import RMSprop, Adam

dir_path = os.path.dirname(os.path.realpath(__file__))

class StateModel:

    def __init__(self, do_load_model=False):

        self.model_path = os.path.join(dir_path, "model.h5")
        self.classes = [x for x in os.listdir(os.path.join(dir_path, "images", "training"))]
        self.checkpoint_cb = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
        self.model = load_model(self.model_path) if do_load_model else self.build_model()

    def build_model(self):
 
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(84, 84, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(len(self.classes)))
        model.add(Activation('softmax'))

        optimizer = Adam(lr=0.00001, decay=8e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def preprocess(self, pil_image):
        img = pil_image.resize((84, 84), ANTIALIAS)
        img = img.convert('RGB')
        return np.array(img)

    def train_epoch(self, X, Y, shuffle=True):

        if shuffle:
            idxes = [x for x in range(len(X))]
            random.shuffle(idxes)
            X_NEW = []
            Y_NEW = []

            for i in idxes:
                X_NEW.append(X[i])
                Y_NEW.append(Y[i])

            X = np.array(X_NEW)
            Y = np.array(Y_NEW)

        acc = self.model.fit(
            X,
            Y,
            batch_size=8,
            epochs=300,
            verbose=2,
            callbacks=[self.checkpoint_cb],
            validation_split=0.4
        )

        return acc

    def predict(self, X):
        answer = self.model.predict(np.array([X]))
        idx = np.argmax(answer)
        return self.classes[idx]
