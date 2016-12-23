import os, h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from vgg16 import vgg16

#
# Load model
#
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

#
# Create and save train
# and validation data generators
#
datagen = ImageDataGenerator(rescale=1./255)

VALIDATE_DATA_DIR = "data/validate"
VALIDATE_FEATURES = "data/intermediate/bottleneck_features_validate.npy"
N_VALIDATE_SAMPLES = 800

validate_generator = datagen.flow_from_directory(
            VALIDATE_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=16,
            class_mode=None,
            shuffle=False)
bottleneck_features_validation = model.predict_generator(validate_generator, N_VALIDATE_SAMPLES)
np.save(open(VALIDATE_FEATURES, "wb"), bottleneck_features_validation)

TRAIN_DATA_DIR = "data/train"
TRAIN_FEATURES = "data/intermediate/bottleneck_features_train.npy"
N_TRAIN_SAMPLES = 2000

train_generator = datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=16,
            class_mode=None,
            shuffle=False)
bottleneck_features_train = model.predict_generator(train_generator, N_TRAIN_SAMPLES)
np.save(open(TRAIN_FEATURES, "wb"), bottleneck_features_train)







