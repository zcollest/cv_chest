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

file_name = './sample_labels.csv'

# Prepare file directories and image paths
if user == "zach":
	os.chdir('/Users/zacharycollester/Documents/')
	PATH = 'cv_chest/data/sample/'
	SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
	images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

	# Load labels
	labels = pd.read_csv(os.path.join(PATH, 'sample_labels.csv'))
elif user == "angelo":
	os.chdir('/home/angelo/Desktop/cv_chest/archive')
	PATH = 'data/sample/'
	SOURCE_IMAGES = os.path.join(PATH, 'sample', 'images')
	images = glob(os.path.join(SOURCE_IMAGES, "*.png"))
	file_name = './sample_labels.csv'

def process_images(width,height):
    """
    1. converts images to gray scale
    2. resizes images
    3. stores data in list of np arrays
    """
    images = pd.read_csv(file_name)
    x = [] # images as arrays
    for img in images:
        full_size_image = cv.imread(img)
        grey = cv.cvtColor(full_size_image, cv.COLOR_BGR2GRAY)
        x.append(cv.resize(grey, (width,height), interpolation=cv.INTER_CUBIC))
    return x	

x = process_images(128,128)
