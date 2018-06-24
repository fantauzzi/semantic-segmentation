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
from tensorflow.python.keras.backend import get_session, set_session
from tensorflow.python.keras.optimizers import Adagrad, Adam
from pathlib import Path
from scipy.misc import imread, imresize, toimage, imsave
from sklearn.model_selection import train_test_split
import numpy as np

# session = get_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
set_session(session)

dataset_path = Path('/home/fanta/datasets/data_road')
training_path = dataset_path / 'training/image_2'
testing_path = dataset_path / 'testing/image_2'
gt_path = dataset_path / 'training/gt_image_2'

n_classes = 2
input_shape = (224, 224, 3)
encoder = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
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
        assert layer.strides == (2, 2)
        # This is a temporary solution, will have to replace upsampling with SegNet upsampling
        x = UpSampling2D(size=layer.pool_size,
                         name=layer.name + '_dec')(x)
    elif type(layer) == Conv2D:  # Convolution -> same convolution, but with no activation function
        x = Conv2D(filters=layer.filters,
                   kernel_size=layer.kernel_size,
                   activation=None,
                   padding=layer.padding,
                   kernel_initializer='glorot_normal',
                   name=layer.name + '_dec')(x)
    else:
        # Otherwise it must be an Input layer from the encoder, which has no corresponding layer in the decoder
        assert type(layer) == InputLayer

x = Conv2D(filters=n_classes,
           kernel_size=(1, 1),
           activation='softmax',
           kernel_initializer='glorot_normal',
           name='softmax_classifier')(x)  # I hope softmax actually operates along the last tensor dimension
model = tf.keras.Model(encoder.input, x)
model.summary()


def get_gt_file_name(file_name):
    underscore_pos = file_name.find('_')
    road_name = file_name[:underscore_pos] + '_road' + file_name[underscore_pos:]
    lane_name = file_name[:underscore_pos] + '_lane' + file_name[underscore_pos:]
    return road_name, lane_name


def load_dataset(images_path, gt_images_path):
    n_images = sum(1 for _ in images_path.glob('*.png'))

    '''
    Load all training images into train_X, and the corresponding images with ground truth into train_Y
    '''
    X = np.zeros(shape=(n_images, input_shape[0], input_shape[1], input_shape[2]))
    Y = np.zeros(shape=(n_images, input_shape[0], input_shape[1], n_classes))
    paths = np.empty(n_images, dtype=np.str)

    # For every image in the dataset...
    for idx, image_path in enumerate(images_path.glob('*.png')):
        # ... load the image and add it to X
        image = imresize(imread(image_path),
                         (input_shape[0], input_shape[1]))  # TODO try with different interpolations, also for the GT
        X[idx, :, :, :] = image
        # Find the file name of the image with the corresponding ground truth
        gt_image_name, _ = get_gt_file_name(image_path.resolve().name)
        # Compose the full path to the ground-truth image
        gt_image_path = gt_images_path / gt_image_name
        # Load the ground truth image and add it to Y (1-hot encoded)
        gt_image = imresize(imread(gt_image_path), (input_shape[0], input_shape[1]))
        gt_image = gt_image[:, :, 2]
        gt_image[gt_image >= 128] = 1  # TODO is this a good idea?
        gt_image[gt_image < 128] = 0
        Y[idx, :, :, 1] = gt_image
        Y[idx, :, :, 0] = 1 - Y[idx, :, :, 1]
        paths[idx] = image_path

    return X, Y, paths


def split_dataset_with_paths(X, Y, paths, train_size, shuffle=True):
    assert len(X) == len(Y) == len(paths)
    assert 0 <= train_size <= 1
    permutations = np.random.permutation(len(paths)) if shuffle else range(len(paths))
    n_train = round(len(paths) * train_size)
    X_shuffled, Y_shuffled, paths_shuffled = X[permutations], Y[permutations], paths[permutations]
    X_train = X_shuffled[: n_train]
    Y_train = Y_shuffled[: n_train]
    paths_train = paths_shuffled[: n_train]
    X_test = X_shuffled[n_train:]
    Y_test = Y_shuffled[n_train:]
    paths_test = paths_shuffled[n_train:]
    return {'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'paths_train': paths_train,
            'paths_test': paths_test}


X, Y, image_paths = load_dataset(training_path, gt_path)
X = X/255
print('Loaded {} training images'.format(X.shape[0]))

split = split_dataset_with_paths(X=X, Y=Y, paths=image_paths, train_size=.8)

X_train = split['X_train']
Y_train = split['Y_train']
X_val = split['X_test']
Y_val = split['Y_test']
paths_train = split['paths_train']
paths_val = split['paths_test']

# TODO add data augmentation
# TODO pre-process and normalize input images (what color space is best?)

optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.fit(x=X_train,
          y=Y_train,
          validation_data=(X_val, Y_val),
          batch_size=2,
          epochs=1,
          verbose=1,
          shuffle=True)

X_test, Y_test, _ = load_dataset(training_path, gt_path)
print('Loaded {} test images'.format(X_test.shape[0]))

Y_pred = model.predict(x=X_test, batch_size=2, verbose=0)

for idx, (x_test, y_pred) in enumerate(zip(X_test, Y_pred)):
    image_mask = (y_pred[:, :, 1] > .5) * 1
    x_test[:, :, 0] = x_test[:, :, 0] + image_mask * 255
    x_test = np.minimum(x_test, np.ones_like(x_test) * 255)
    imsave('output/' + str(idx) + '.png', x_test)

session.close()  # Closing the session prevents a non-systematic error at the end of program execution
