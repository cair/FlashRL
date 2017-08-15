import numpy as np
from PIL.Image import ANTIALIAS
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.contrib.keras.python.keras.models import Sequential, load_model
import os

from tensorflow.contrib.keras.python.keras.optimizers import RMSprop

dir_path = os.path.dirname(os.path.realpath(__file__))

class StateModel:

    def __init__(self, do_load_model=False):

        self.model_path = os.path.join(dir_path, "model.h5")
        self.classes = [x for x in os.listdir(os.path.join(dir_path, "images"))]
        self.checkpoint_cb = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
        self.model = load_model(self.model_path) if do_load_model else self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
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
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.classes)))
        model.add(Activation('relu'))

        optimizer = RMSprop(lr=0.00001, decay=8e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def preprocess(self, pil_image):
        img = pil_image.resize((150, 150), ANTIALIAS)
        img = img.convert('RGB')
        return np.array(img)

    def train_epoch(self, X, Y):
        acc = self.model.fit(
            X,
            Y,
            batch_size=32,
            epochs=5000,
            verbose=2,
            callbacks=[self.checkpoint_cb]
        )

        return acc

    def predict(self, X):
        answer = self.model.predict(np.array([X]))
        idx = np.argmax(answer)
        return self.classes[idx]
