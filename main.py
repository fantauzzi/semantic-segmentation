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
config.gpu_options.allow_growth=True
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
        '''
        Set the activation function to None, unless the layer is the last in the decoder (corresponding to the first in the encoder), then set it to softmax
        '''
        # activation = 'softmax' if idx == n_encoder_layers-2 else None

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


# Load the first image in the dataset to discover the images resolution

# first_image_path = next(training_path.glob('*.png'))
# first_image = imread(first_image_path)
# image_res_y, image_res_x, image_res_depth = first_image.shape

def load_dataset(images_path, gt_images_path):
    n_images = sum(1 for _ in images_path.glob('*.png'))

    '''
    Load all training images into train_X, and the corresponding images with ground truth into train_Y
    '''
    X = np.zeros(shape=(n_images, input_shape[0], input_shape[1], input_shape[2]))
    Y = np.zeros(shape=(n_images, input_shape[0], input_shape[1], n_classes))

    # For every image in the dataset...
    for idx, image_path in enumerate(images_path.glob('*.png')):
        # ... load the image and add it to X
        image = imresize(imread(image_path), (input_shape[0], input_shape[1]))  # TODO try with different interpolations, also for the GT
        X[idx, :, :, :] = image
        # Find the file name of the image with the corresponding ground truth
        gt_image_name, _ = get_gt_file_name(image_path.resolve().name)
        # Compose the full path to the ground-truth image
        gt_image_path = gt_images_path / gt_image_name
        # Load the ground truth image and add it to Y (1-hot encoded)
        gt_image = imresize(imread(gt_image_path), (input_shape[0], input_shape[1]))
        gt_image = gt_image[:, :, 2]
        gt_image[gt_image>=128] = 1  # TODO is this a good idea?
        gt_image[gt_image < 128] = 0
        Y[idx, : , :, 1] = gt_image
        Y[idx, :, :, 0] = 1 - Y[idx, : , :, 1]

    return X, Y

X, Y = load_dataset(training_path, gt_path)
print('Loaded {} training images'.format(X.shape[0]))


# TODO add data augmentation
# TODO pre-process and normalize input images (what color space is best?)

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, shuffle=True, train_size=.8, test_size=.2)

optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.fit(x=X, y=Y, batch_size=2, epochs=1, verbose=1, validation_split=.2, shuffle=True)

X_test, Y_test = load_dataset(training_path, gt_path)
print('Loaded {} test images'.format(X_test.shape[0]))

Y_pred = model.predict(x=X_test, batch_size=2, verbose=0)

for idx, (x_test, y_pred) in enumerate(zip(X_test, Y_pred)):
    image_mask = (y_pred[:, :, 1] > .5)*1
    x_test[:, :, 0] = x_test[:, :, 0] + image_mask * 64
    x_test = np.minimum(x_test, np.ones_like(x_test)*255)
    # image_mask = np.stack((np.zeros((224,224)), image_mask*255, np.zeros((224,224)), np.ones((224, 224))*128), -1)
    # x_test = np.append(x_test, np.ones((224,224,1))*255, -1)
    # image = toimage(x_test, mode='RGBA')
    # image.paste(image_mask, box=None, mask=image_mask)
    imsave('output/'+str(idx)+'.png', x_test)



"""im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
mask = scipy.misc.toimage(mask, mode="RGBA")
street_im = scipy.misc.toimage(image)
street_im.paste(mask, box=None, mask=mask)

yield os.path.basename(image_file), np.array(street_im)"""

session.close()  # Closing the session prevents a non-systematic error at the end of program execution
