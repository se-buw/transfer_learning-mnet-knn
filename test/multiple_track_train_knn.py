import numpy as np
import pandas as pd
import cv2
import os
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle 
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns

IMAGE_SIZE = 224

df1 = pd.read_csv('./recources/2023-01-18_09-53-06-514944.csv', 
                  names=["ID", "label"])

df2 = pd.read_csv('./recources/2023-01-18_10-53-41-947901.csv', 
                  names=["ID", "label"])    

df3 = pd.read_csv('./recources/2023-01-18_11-17-27-856241.csv', 
                  names=["ID", "label"])

df4 = pd.read_csv('./recources/2023-01-18_11-31-50-113593.csv', 
                  names=["ID", "label"])       


# Pre processing 
train_images = []
train_labels = []
# Loading the images
for img_name, label in zip(df1.ID, df1.label):
  img = cv2.imread(os.path.join('./recources/img_2023-01-18_09-53-06-514944/', img_name))
  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  if img is not None:
    train_images.append(img)
    train_labels.append(label)

for img_name, label in zip(df2.ID, df2.label):
  img = cv2.imread(os.path.join('./recources/img_2023-01-18_10-53-41-947901/', img_name))
  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  if img is not None:
    train_images.append(img)
    train_labels.append(label)

for img_name, label in zip(df3.ID, df3.label):
  img = cv2.imread(os.path.join('./recources/img_2023-01-18_11-17-27-856241/', img_name))
  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  if img is not None:
    train_images.append(img)
    train_labels.append(label)

for img_name, label in zip(df4.ID, df4.label):
  img = cv2.imread(os.path.join('./recources/img_2023-01-18_11-31-50-113593/', img_name))
  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  if img is not None:
    train_images.append(img)
    train_labels.append(label)


# Feature extraction with MobileNetV2
model = MobileNetV3Large(weights='imagenet',include_top=False)
model_feature_list = []

for img in train_images:
  img_data = np.expand_dims(img, axis=0)
  img_data = preprocess_input(img_data)
  model_feature = model.predict(img_data)
  print('Shape:', np.shape(model_feature))
  print('Size:', model_feature.size)
  model_feature_np = np.array(model_feature)
  model_feature_list.append(model_feature_np.flatten())


train_x = np.array(model_feature_list)
print(np.shape(train_x))

# Label encoding
train_y = pd.Series(train_labels)
label_encode = {'left': 0, 'right': 1, 'forward':2}

train_y = train_y.replace(label_encode)

print(train_y)


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

print('--------- Train------------')
print(sklearn.metrics.classification_report(y_train,y_pred_train))
print('--------- Test------------')
print(sklearn.metrics.classification_report(y_test,y_pred_test))


# Confusion Matrix for training data
cm=sklearn.metrics.confusion_matrix(y_test, y_pred_test)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,fmt='g')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(['Covid', 'Normal'])
# ax.yaxis.set_ticklabels(['Covid','Normal'])
plt.show()


classifier.fit(train_x,train_y)

# Save the model
knnPickle = open('./recources/trained_model', 'wb') 
pickle.dump(classifier, knnPickle)  
# close the file
knnPickle.close()

