#
# Filter visualization

from vgg16 import vgg16
import h5py
from keras import backend as K
import numpy as np
from scipy.misc import imsave, imread

# Load vgg16 model weights

WEIGHTS_PATH = "models/vgg16_weights.h5"
IMG_HEIGHT, IMG_WIDTH = 499, 429
model = vgg16(IMG_HEIGHT, IMG_WIDTH)
with h5py.File(WEIGHTS_PATH) as f:
    for layer in range(f.attrs["nb_layers"]):
        if layer >= len(model.layers):
            break
        layer_weights = f["layer_{0}".format(layer)]
        weights = [ layer_weights["param_{0}".format(param)]\
            for param in range(layer_weights.attrs["nb_params"]) ]
        model.layers[layer].set_weights(weights)

print "Model weights loaded."

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype("uint8")
    return x

layers = { layer.name: layer for layer in model.layers }

LAYER_NAME = "conv5_1"
first_layer = model.layers[0]
step = 1e-2
image_path = "dog.1014.jpg"

for filter_index in [31]:# [0, 31, 63, 127, 255, 511]:
    input_img = first_layer.input
    layer_output = layers[LAYER_NAME].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    grads = K.gradients(loss, input_img)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.array(imread(image_path)) # np.random.random((1, 3, IMG_WIDTH, IMG_HEIGHT)) * 20 + 128.
    input_img_data = input_img_data.reshape((1, 3, IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)

    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave("images/dog_{0}_filter_{1}.png".format(LAYER_NAME, filter_index), img)