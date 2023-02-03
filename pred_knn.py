import pickle 
import numpy as np
import cv2
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet import preprocess_input

IMAGE_SIZE = 224
MODEL_PATH = './recources/trained_model'
  
# load the model from disk
try: 
  loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
except:
  print("File not found. Check the MODEL_PATH!")
  exit()


model = MobileNetV3Large(weights='imagenet', include_top=False)

def predict_direction(img):
    try:
      img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
      img_data = np.expand_dims(img, axis=0)
      img_data = preprocess_input(img_data)
      model_feature = model.predict(img_data)
      model_feature_np = np.array(model_feature).flatten()
      #model_feature_list.append(model_feature_np.flatten())
      x_test = np.array(np.expand_dims(model_feature_np, axis=0))
      result = loaded_model.predict(x_test) 
      return result
    except:
       print("{0} not found or corrupted.".format(img))