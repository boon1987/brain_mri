import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# make_edges reads an image from /pfs/images and outputs the result of running edge detection on that image to /pfs/out.
# Note that /pfs/images and /pfs/out are special directories that Pachyderm injects into the container.
input_data_path = "/pfs/pipeline_input_data"
pach_std_output_path = "pfs/out"
output_data_path = os.path.join(pach_std_output_path, "mri_edges")
os.makedirs(output_data_path, exist_ok=True)


def make_edges(image, output_data_path):
   img = cv2.imread(image)
   edges = cv2.Canny(img,100,200)
   edges = cv2.resize(edges, (edges.shape[1]*4, edges.shape[0]*4))
   tail = os.path.split(image)[1]
   plt.imsave(os.path.join(output_data_path, os.path.splitext(tail)[0]+'.png'), edges, cmap = 'gray')


# walk /pfs/pipeline_input_data and call make_edges on every file found
output_list = []
counter=0
for dirpath, dirs, files in os.walk(input_data_path):
   #print(dirpath)
   for file in files:
      filepath = os.path.join(dirpath, file)
      output_list.append([filepath])
      #print(filepath)
      # if counter == 10:
      #    break
      if filepath.split(".")[-1] == "tif":
         print('process file ', filepath)
         #make_edges(filepath, output_data_path)
         counter=counter+1
print("number of images being processed: ", counter)


# with open("/pfs/pipeline_input_data/kaggle_3m_dataset/data.csv", 'r') as file:
#   csvreader = csv.reader(file)
#   for row in csvreader:
#     print(row)
    
    
# df = pd.read_csv("/pfs/pipeline_input_data/kaggle_3m_dataset/data.csv")
# print(df.head())
# with open(os.path.join(output_data_path,"data.pickle"), 'wb') as f:
#    pickle.dump(output_list, f)
   
with open(os.path.join(pach_std_output_path,"data.pickle"), 'wb') as f:
   pickle.dump(output_list, f)