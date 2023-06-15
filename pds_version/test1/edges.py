import cv2
import csv
import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle

# make_edges reads an image from /pfs/images and outputs the result of running
# edge detection on that image to /pfs/out. Note that /pfs/images and
# /pfs/out are special directories that Pachyderm injects into the container.
# def make_edges(image):
#    img = cv2.imread(image)
#    tail = os.path.split(image)[1]
#    edges = cv2.Canny(img,100,200)
#    plt.imsave(os.path.join("/pfs/out", os.path.splitext(tail)[0]+'.png'), edges, cmap = 'gray')

# walk /pfs/pipeline_input_data and call make_edges on every file found
output_list = []
counter=0
for dirpath, dirs, files in os.walk("/pfs/pipeline_input_data"):
   #print(dirpath)
   for file in files:
      filepath = os.path.join(dirpath, file)
      #print(filepath)
      break
      # if counter == 10:
      #    break
      print(file)
      output_list.append([dirpath, dirs, files])
      counter=counter+1
      #make_edges(os.path.join(dirpath, file))

with open("/pfs/pipeline_input_data/kaggle_3m_dataset/data.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    print(row)
    
# df = pd.read_csv("/pfs/pipeline_input_data/kaggle_3m_dataset/data.csv")
# print(df.head())
# with open(os.path.join("/pfs/out","data.pickle"), 'wb') as f:
#    pickle.dump(output_list, f)