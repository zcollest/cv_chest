import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import os
from glob import glob

# Prepare file directories and image paths
os.chdir('/Users/zacharycollester/Documents/')
PATH = 'cv_chest/data/sample/'
SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

# Load labels
labels = pd.read_csv(os.path.join(PATH, 'sample_labels.csv'))


def process_images():
    """
    1. converts images to gray scale
    2. resizes images
    3. stores data in list of np arrays
    """
    x = [] # images as arrays
    WIDTH = 128
    HEIGHT = 128
    for img in images:
        full_size_image = cv.imread(img)
        grey = cv.cvtColor(full_size_image, cv.COLOR_BGR2GRAY)
        x.append(cv.resize(grey, (WIDTH,HEIGHT), interpolation=cv.INTER_CUBIC))
    return x

x = process_images()


cv.waitKey(0)
cv.destroyAllWindows()
