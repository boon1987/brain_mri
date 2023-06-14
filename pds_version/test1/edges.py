import cv2
import numpy as np
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

# walk /pfs/images and call make_edges on every file found
output_list = []
counter=0
for dirpath, dirs, files in os.walk("/pfs/pipeline_input_data"):
   for file in files:
      if counter == 10:
         break
      print(file)
      output_list.append([dirpath, dirs, files])
      counter=counter+1
      #make_edges(os.path.join(dirpath, file))
      
with open(os.path.join("/pfs/out","data.pickle"), 'wb') as f:
   pickle.dump(output_list, f)