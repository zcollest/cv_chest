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

### new imports
from imblearn.under_sampling import ClusterCentroids

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
else:
    file_name = 'sample_labels.csv'
    os.chdir('C:/Users/Angelo/Desktop/archive/sample')

def process_images(width,height):
    """
    1. resizes images
    2. stores data in dictionary 
        - key is image ID
        - values include resized image arrays and class labels (both primary and subclass labels)
    """

    ##################################
    # MAYBE EXTRA QUALITY CHECK:
    # REMOVE IMAGES THAT ARE TOO WHITE OR TO DARK
    ################################## 

    #read file
    samples = pd.read_csv(file_name)
    #get images names
    images = samples['Image_Index']
    x = {} # images as arrays
    #iterate through the names and populate dictionary with the class and the 1-d representation of each image
    for i, img in enumerate(images):
        print(i,'/',len(images),end='\r')
        #full_size_image = cv.imread(os.path.join("./",SOURCE_IMAGES,img.split('/')[-1]))
        og_image = cv.imread(os.path.join('images',img))
        x[img] = {}
        reshaped_resized_image = np.reshape(cv.resize(og_image, (width,height), interpolation=cv.INTER_CUBIC),-1)
        x[img]['array'] = reshaped_resized_image
        x[img]['class'] = samples.iloc[i]['Finding_Labels'].split('|')[0]
        if len(samples.iloc[i]['Finding_Labels'].split('|')) > 1:
            x[img]['subclass'] = samples.iloc[i]['Finding_Labels'].split('|')[1:]
    #return dictionary
    return x
    

# visualizing class disitributions for samples
#nosubclass = [(x, sampledict[x]["class"]) for x in sampledict if 'subclass' not in sampledict[x]]
#nosubclass = pd.DataFrame(nosubclass)
#nosubclasshist = nosubclass.iloc[:,1].hist(figsize = [60,20])
#nosubclass.iloc[:,1].value_counts()

def downsample(minimum_elements = 100,mode = 'centroids'):
    '''
        This function allows to downsample the dataset.
        The minimum_elements parameter specifies the minimum number of elements that is allowed for each class. If a class has less than 'minimum_elements' element it is discarded.
        mode (string) is the algorithm we want to use for the downsampling procedure.
    '''
    #use process_images function to process the images. I put 32,32 as width and height because I was scared of how long it could take if I would consider bigger pictures, but we can definitely test it
    sampledict = process_images(32,32)
    #define dataset as a dataset that only contains mono labeled images 
    nosubclass = [(x, sampledict[x]["class"]) for n,x in enumerate(sampledict) if 'subclass' not in sampledict[x]]
    #create dataframe out of dataset
    nosubclass_df = pd.DataFrame(nosubclass)
    #rename class column for better handling
    nosubclass_df = nosubclass_df.rename(columns={1: "target"})
    #get indexes and values of the value counts for the classes
    indexes_values = zip(nosubclass_df.target.value_counts().index,nosubclass_df.target.value_counts())
    #remove from dataframe all those classes that have less than 'minimum_elements' entries
    for index,value in indexes_values:
        if int(value) < minimum_elements:
            nosubclass_df = nosubclass_df[nosubclass_df.target != index]
    #define dataset for undersampling
    X = [list(sampledict[x]['array']) for x in list(nosubclass_df[0])]
    #trasform dataset into dataframe
    X = pd.DataFrame(X)
    
    
    #map target to integer. Change classes from labels to integer indexes
    target_to_integer = {} #dictionary for mapping classes strings to classes integer index
    for integer,target in enumerate(nosubclass_df.target.unique()):
        target_to_integer[target] = integer
    targets = []
    for target in nosubclass_df.target:
        targets.append(target_to_integer[target])
    #substitute classes in target column in dataframe with their integer indexes 
    nosubclass_df.target = targets
    
    #visualize distribution (uncomment if you want to use it)
    nosubclass.target.value_counts().plot(kind='bar', title='Count (target)')
    
    #undersample by centroids
    if not mode or mode == 'centroids':
        #the ClusterCentroids functions creates a CLusterCentroids object. It provides the function fit_sample which by default will downsample our dataset by means
        #of the KNN algorithm
        cc = ClusterCentroids()
        X_cc, y_cc = cc.fit_sample(X,nosubclass_df.target)
    
    return X_cc, y_cc

X_cc,y_cc = downsample()

