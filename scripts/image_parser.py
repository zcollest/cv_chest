import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import os
from glob import glob
import sys
import argparse

my_parser = argparse.ArgumentParser(description='image parser')
my_parser.add_argument('--user',
                       metavar='user',
                       type=str,
                       help='Who is using the script')
args = my_parser.parse_args()

user = args.user

# Prepare file directories and image paths
if user == "zach":
    os.chdir('/Users/zacharycollester/Documents/')
    PATH = 'cv_chest/data/sample/'
    SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
    file_name = os.path.join(PATH, "sample_labels_all.csv")
elif user == "angelo":
	os.chdir('/home/angelo/Desktop/cv_chest/archive')
	PATH = 'data/sample/'
	SOURCE_IMAGES = os.path.join(PATH, 'sample', 'images')

def process_images(width,height):
    """
    1. resizes images
    2. stores data in dictionary 
        - key is image ID
        - values include resized image arrays and class labels (both primary and subclass labels)
    """
    samples = pd.read_csv(file_name)
    images = samples['Image_Index']
    x = {} # images as arrays
    for i, img in enumerate(images):
        #full_size_image = cv.imread(os.path.join("./",SOURCE_IMAGES,img.split('/')[-1]))
        #grey = cv.cvtColor(full_size_image, cv.COLOR_BGR2GRAY)
        x[img] = {}
        #x[img]['array'] = cv.resize(full_size_image, (width,height), interpolation=cv.INTER_CUBIC)
        x[img]['class'] = samples.iloc[i]['Finding_Labels'].split('|')[0]
        if len(samples.iloc[i]['Finding_Labels'].split('|')) > 1:
            x[img]['subclass'] = samples.iloc[i]['Finding_Labels'].split('|')[1:]
    return x	
sampledict = process_images(128,128)

# visualizing class disitributions for samples
nosubclass = [(x, sampledict[x]["class"]) for x in sampledict if 'subclass' not in sampledict[x]]
nosubclass = pd.DataFrame(nosubclass)
nosubclasshist = nosubclass.iloc[:,1].hist(figsize = [60,20])
nosubclass.iloc[:,1].value_counts()

