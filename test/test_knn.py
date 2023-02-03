import numpy as np
import pandas as pd
import os
import cv2
from sklearn.metrics import confusion_matrix
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,roc_curve,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold


normal_images=os.listdir('./recources/img')
#print(normal_images)

# Empty DataFrame
normal_image_df=pd.DataFrame()

normal_image_df['ID']=normal_images
normal_image_df['label']='fwd'


# print(normal_image_df)

train_images = []
train_labels = []

for img_name, label in zip(normal_image_df.ID, normal_image_df.label):
  img = cv2.imread(os.path.join('./recources/img', img_name))
  img = cv2.resize(img, (224, 224))
  if img is not None:
    train_images.append(img)
    train_labels.append('fwd')



model = MobileNetV2(weights='imagenet', include_top=False)
model_feature_list = []

for img in train_images:
  img_data = np.expand_dims(img, axis=0)
  img_data = preprocess_input(img_data)
  model_feature = model.predict(img_data)
  model_feature_np = np.array(model_feature)
  model_feature_list.append(model_feature_np.flatten())

train_x = np.array(model_feature_list)
print(np.shape(train_x))
print(train_x.size)

train_y = pd.Series(train_labels)
label_encode = {'fwd': 0}

train_y = train_y.replace(label_encode)

#print(train_y)


classifier=KNeighborsClassifier(n_neighbors=120,p=2)
classifier.fit(train_x,train_y)

# # Hyperparameter Tuning
param_grid = {'n_neighbors':[80,90,120,200]}
grid=GridSearchCV(estimator=KNeighborsClassifier(),cv=5,param_grid=param_grid)
grid.fit(train_x,train_y)


# Predicting the Train Set Results
y_pred_train=classifier.predict(train_x)

print(y_pred_train)