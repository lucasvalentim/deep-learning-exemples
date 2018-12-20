import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.preprocessing.image import img_to_array


def prepare_image(image):
    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((28, 28))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255

    return image

def decode_predictions(preds):
    return {
        'label': str(np.argmax(preds)),
        'probability': float(np.max(preds))
    }

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1,1)))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def load_production_model():
    model = build_model()

    weights_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights.hdf5')

    model.load_weights(weights_filepath)
    
    return model
