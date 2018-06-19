import tensorflow as tf
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D

model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model.summary()


def clone_layer(layer):
    layer_type = type(layer)
    if layer_type == Input:
        clone = Input()


for layer in model.layers:
    if type(layer) == Conv2D:
        print('Convolution!')


x = model.output
enc_dec = tf.keras.Model(model.input, x)


print('Done! ')