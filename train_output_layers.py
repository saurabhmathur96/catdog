import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense

N_TRAIN_SAMPLES = 2000
TRAIN_FEATURES = "data/intermediate/bottleneck_features_train.npy"
with open(TRAIN_FEATURES) as f:
    train_data = np.load(f)
    train_labels = np.array([0] * (N_TRAIN_SAMPLES / 2) + [1] * (N_TRAIN_SAMPLES / 2))



N_VALIDATE_SAMPLES = 800
VALIDATE_FEATURES = "data/intermediate/bottleneck_features_validate.npy"
with open(VALIDATE_FEATURES) as f:
    validation_data = np.load(f)
    validation_labels = np.array([0] * (N_VALIDATE_SAMPLES / 2) + [1] * (N_VALIDATE_SAMPLES / 2))

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

N_EPOCH = 50
model.fit(train_data, train_labels,
            nb_epoch=N_EPOCH, batch_size=16,
            validation_data=(validation_data, validation_labels))

TOP_MODEL_WEIGHTS_PATH = "models/top_model.h5"
model.save_weights(TOP_MODEL_WEIGHTS_PATH)