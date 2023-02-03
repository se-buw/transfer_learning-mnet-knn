import os
import cv2
import pickle 
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from keras.applications import MobileNetV3Large
from keras.applications.mobilenet import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

IMAGE_SIZE = 224
IMAGE_PATH = "./recources/2023-02-03_12-39-54-403937.csv"

# Reading the captured images and their direction, saved during the race
try:
  normal_image_df = pd.read_csv(IMAGE_PATH, 
                    names=["ID", "label"])
except:
  print("File not found. Change the IMAGE_PATH!")
  exit()

# Pre processing 
train_images = []
train_labels = []

# Loading the images
for img_name, label in zip(normal_image_df.ID, normal_image_df.label):
  try:
    img = cv2.imread(os.path.join('./recources/img/', img_name))
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if img is not None:
      train_images.append(img)
      train_labels.append(label)
  except: 
    print("{0} not found or corrupted.".format(img_name))

# Feature extraction with MobileNetV2
model = MobileNetV3Large(weights='imagenet',include_top=False)
model_feature_list = []

for img in train_images:
  img_data = np.expand_dims(img, axis=0)
  img_data = preprocess_input(img_data)
  model_feature = model.predict(img_data)
  model_feature_np = np.array(model_feature)
  model_feature_list.append(model_feature_np.flatten())


train_x = np.array(model_feature_list)

# Label encoding
train_y = pd.Series(train_labels)
label_encode = {'left': 0, 'right': 1, 'forward':2}

train_y = train_y.replace(label_encode)


# Building KNN classifier
""""
p=2: manhattan_distance (l1)
p=2: euclidean_distance (l2)
"""
classifier=KNeighborsClassifier(n_neighbors=5,p=2) 
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.20, random_state=42)

classifier.fit(X_train,y_train)


# Hyperparameter Tuning
# param_grid = {'n_neighbors':[80,90,120,200]}
# grid=GridSearchCV(estimator=KNeighborsClassifier(),cv=5,param_grid=param_grid)
# grid.fit(train_x,train_y)

y_pred_train=classifier.predict(X_train)
y_pred_test=classifier.predict(X_test)
try:
  print('--------- Train------------')
  print(sklearn.metrics.classification_report(y_train,y_pred_train))
  print('--------- Test------------')
  print(sklearn.metrics.classification_report(y_test,y_pred_test))
except:
  print('Error: Train or Test set are too small. OR did not found all the classes.')


try:
  # Confusion Matrix for training data
  cm=sklearn.metrics.confusion_matrix(y_test, y_pred_test)
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, ax = ax,fmt='g')
  # labels, title and ticks
  ax.set_xlabel('Predicted Direction')
  ax.set_ylabel('Actual Direction')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels(['Forward', 'Left', 'Right'])
  ax.yaxis.set_ticklabels(['Forward', 'Left', 'Right'])
  plt.show()
except:
  print('Error: Train or Test set are too small. OR did not found all the classes.')


classifier.fit(train_x,train_y)

# Save the model
knnPickle = open('./recources/trained_model', 'wb') 
pickle.dump(classifier, knnPickle)  
# close the file
knnPickle.close()

