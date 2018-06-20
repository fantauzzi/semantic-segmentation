import tensorflow as tf
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.layers import UpSampling2D
from tensorflow.python.keras._impl.keras.engine.input_layer import InputLayer
from tensorflow.python.keras._impl.keras.layers import Softmax
from tensorflow.python.keras.backend import get_session

n_classes = 10
encoder = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = encoder.output

encoder.summary()

'''
Scan the layers in the encoder in reverse order, and build the decoder, adding the corresponding layers to the model.
'''
# TODO add batch normalization
last_layer = True
n_encoder_layers = len(encoder.layers)
for idx, layer in enumerate(reversed(encoder.layers)):
    if type(layer) == MaxPooling2D:  # Max pooling -> upsampling
        # To simplify computation of the UpSampling2D size, I assume that layer.strides is (2,)
        assert layer.strides ==  (2,2)
        # This is a temporary solution, will have to replace upsampling with SegNet upsampling
        x = UpSampling2D(size=layer.pool_size,
                         name=layer.name+'_dec')(x)
    elif type(layer) == Conv2D:  # Convolution -> same convolution, but with no activation function
        '''
        Set the activation function to None, unless the layer is the last in the decoder (corresponding to the first in the encoder), then set it to softmax
        '''
        # activation = 'softmax' if idx == n_encoder_layers-2 else None

        x = Conv2D(filters=layer.filters,
                   kernel_size=layer.kernel_size,
                   activation=None,
                   padding=layer.padding,
                   name=layer.name+'_dec')(x)
    else:
        # Otherwise it must be an Input layer from the encoder, which has no corresponding layer in the decoder
        assert type(layer) == InputLayer

x = Conv2D(filters=n_classes,
           kernel_size=(1,1),
           activation='softmax',
           name='softmax_classifier')(x)  # I hope softmax actually operates along the last tensor dimension
model = tf.keras.Model(encoder.input, x)
model.summary()

# sess = get_session()
# sess.close()
print('Done! ')