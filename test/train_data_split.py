import os
import shutil
import pandas as pd

# Parent Directory path 
parent_dir = "./recources/training_data/"
    
# Path 
# path = os.path.join(parent_dir, 'left') 
# os.mkdir(path) 
# path = os.path.join(parent_dir, 'right') 
# os.mkdir(path) 
# path = os.path.join(parent_dir, 'forward') 
# os.mkdir(path) 
os.makedirs('./recources/training_data/validation_dir/left')
os.makedirs('./recources/training_data/validation_dir/right')
os.makedirs('./recources/training_data/validation_dir/forward')




df = pd.read_csv('./recources/2023-01-17_11-36-25-728369.csv')

#print(df) 

for index, row in df.iterrows():
  if row[1] == 'left':
    src = './recources/img_3/'+row[0]
    dest = './recources/training_data/validation_dir/left'
    shutil.copy(src, dest)
  elif row[1] == 'right':
    src = './recources/img_3/'+row[0]
    dest = './recources/training_data/validation_dir/right'
    shutil.copy(src, dest)
  elif row[1] == 'forward':
    src = './recources/img_3/'+row[0]
    dest = './recources/training_data/validation_dir/forward'
    shutil.copy(src, dest)
