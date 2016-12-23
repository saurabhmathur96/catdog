import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from vgg16 import vgg16


#
# Load model

WEIGHTS_PATH = "models/vgg16_weights.h5"
IMG_HEIGHT, IMG_WIDTH = 150, 150
model = vgg16(IMG_HEIGHT, IMG_WIDTH)
with h5py.File(WEIGHTS_PATH) as f:
    for layer in range(f.attrs["nb_layers"]):
        if layer >= len(model.layers):
            break
        layer_weights = f["layer_{0}".format(layer)]
        weights = [ layer_weights["param_{0}".format(param)] 
            for param in range(layer_weights.attrs["nb_params"]) ]
        model.layers[layer].set_weights(weights)

print "Model weights loaded."

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation="sigmoid"))

TOP_MODEL_WEIGHTS_PATH = "models/top_model.h5"
top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

print "Top model weights loaded."


model.add(top_model)

for layer in model.layers[:25]:
    layer.trainable = False


model.compile(loss="binary_crossentropy",
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=["accuracy"])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


TRAIN_DATA_DIR = "data/train"
train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="binary")

test_datagen = ImageDataGenerator(rescale=1./255)

VALIDATE_DATA_DIR = "data/validate"
validation_generator = test_datagen.flow_from_directory(
        VALIDATE_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="binary")

# fine-tune the model
N_TRAIN_SAMPLES = 2000
N_VALIDATE_SAMPLES = 800
N_EPOCH = 25
model.fit_generator(
        train_generator,
        samples_per_epoch=N_TRAIN_SAMPLES,
        nb_epoch=N_EPOCH,
        validation_data=validation_generator,
        nb_val_samples=N_VALIDATE_SAMPLES)

FINETUNED_MODEL_PATH = "models/finetuned_vgg16.h5"
model.save(FINETUNED_MODEL_PATH)
