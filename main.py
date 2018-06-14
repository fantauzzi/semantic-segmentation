import tensorflow as tf

model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model.summary()

print('Done!')