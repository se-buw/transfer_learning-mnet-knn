import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.preprocessing import image
from keras.applications import MobileNetV3Large

mnet = MobileNetV3Large(weights='imagenet', include_top=False)


x = mnet.layers
print(mnet.summary())


# print("weights:", len(x.weights))
# print("trainable_weights:", len(x.trainable_weights))
# print("non_trainable_weights:", len(x.non_trainable_weights))

def prepare_image(file):
  # img = tf.keras.utils.load_img(file, target_size=(224,224))
  # img_array = tf.keras.utils.img_to_array(img)
  img = cv2.imread(file)
  img = cv2.resize(img, (224, 224))
  img_array_expanded_dims = np.expand_dims(img, axis=0)
  return  keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


preprocessed_image = prepare_image('./recources/img/2023-01-17_10-32-14-211965.jpg')

predictions = mnet.predict(preprocessed_image)


# for layer in mnet.layers:
#   print(layer.trainable)
#     #layer.trainable = False

print('Shape: {}'.format(np.shape(predictions)))

print(predictions)


output_neuron = np.argmax(predictions[0])
print('Most active neuron: {} ({:.2f}%)'.format(
    output_neuron,
    100 * predictions[0][output_neuron]
))

from keras.applications.mobilenet_v2 import decode_predictions

for name, desc, score in decode_predictions(predictions)[0]:
    print('- {} ({:.2f}%%)'.format(desc, 100 * score))
    