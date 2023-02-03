import os
import tensorflow as tf
from tensorflow import keras
import numpy
from matplotlib import pyplot as plt

base_dir = './recources/training_data'

train_dir = os.path.join(base_dir, 'train_dir')
validation_dir = os.path.join(base_dir, 'validation_dir')

image_size = 224
batch_size = 64

train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(image_size, image_size), batch_size=batch_size)

validation_datagen = keras.preprocessing.image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(image_size, image_size), batch_size=batch_size)

IMG_SHAPE = (image_size, image_size, 3)

base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False)
base_model.trainable = False

base_model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(train_generator.num_classes, activation='sigmoid')
])

base_model.summary()

base_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 100
steps_per_epoch = numpy.ceil(train_generator.n / batch_size)
validation_steps = numpy.ceil(validation_generator.n / batch_size)

history = base_model.fit_generator(generator=train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)


final_loss, final_accuracy = base_model.evaluate(validation_generator, steps = validation_steps)
print('Final loss: {:.2f}'.format(final_loss))
print('Final accuracy: {:.2f}%'.format(final_accuracy * 100))



base_model.save('MobileNet_TransferLearning_Fruits360v48.h5')