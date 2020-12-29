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
from sklearn.decomposition import PCA

### new imports
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

from collections import Counter

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
        gray_image = cv.cvtColor(og_image, cv.COLOR_BGR2GRAY)

        x[img] = {}
        reshaped_resized_image = []
        for pixel in np.reshape(cv.resize(gray_image, (width,height), interpolation=cv.INTER_CUBIC),-1):
            if int(pixel) < 0 or pixel > 255:
                print('ERROR pixel value to low or too high')
            else:
                reshaped_resized_image.append(int(pixel)/255)
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

def sampling_dict(y,n):
    '''
    The function takes a list of labels (y) as input and returns a dictionary.
    The 'n' parameter is the number of samples to be considered for each label. 
    n cannot be lower than 'minimum_elements'
    The dictionary has the labels as keys and the number of samples to consider for each label as value.
    '''
    res = {}
    for key in y:
        res[key] = n
    return res

tomek_min = 100 

def tomek_ratio(y):
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = tomek_min
    return target_stats

def downsample(minimum_elements = 100,mode = 'centroids',undersampling_limit=None,oversampling_limit=None):
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
    nosubclass_df = nosubclass_df.rename(columns={1: "label"})
    #get indexes and values of the value counts for the classes
    indexes_values = zip(nosubclass_df.label.value_counts().index,nosubclass_df.label.value_counts())
    #remove from dataframe all those classes that have less than 'minimum_elements' entries
    for index,value in indexes_values:
        if int(value) < minimum_elements:
            nosubclass_df = nosubclass_df[nosubclass_df.label != index]
    #define dataset for undersampling
    X = [list(sampledict[x]['array']) for x in list(nosubclass_df[0])]
    #trasform dataset into dataframe
    X = pd.DataFrame(X)
    
    
    #map target to integer. Change classes from labels to integer indexes
    label_to_integer = {} #dictionary for mapping classes strings to classes integer index
    for integer,label in enumerate(nosubclass_df.label.unique()):
        label_to_integer[label] = integer
    labels = []
    for label in nosubclass_df.label:
        labels.append(label_to_integer[label])
    #substitute classes in target column in dataframe with their integer indexes 
    nosubclass_df.label = labels
    
    #visualize distribution (uncomment if you want to use it)
    #nosubclass.label.value_counts().plot(kind='bar', title='Count (target)')
    
    #undersample by centroids
    if not mode or mode == 'centroids':
        #the ClusterCentroids functions creates a CLusterCentroids object. It provides the function fit_sample which by default will downsample our dataset by means
        #of the KNN algorithm
        if undersampling_limit and undersampling_limit >= minimum_elements:
            cc = ClusterCentroids(sampling_strategy=sampling_dict(nosubclass_df.label,undersampling_limit))
        else:
            cc = ClusterCentroids()
        return cc.fit_sample(X,nosubclass_df.label)
    elif 'SMOTE' in mode:
        if len(mode.split('+')) == 1:
            print('SMOTE')
            if oversampling_limit:
                #TODO add check: oversampling limit must be higher than number of elements in biggest class
                sm = SMOTE(sampling_strategy=sampling_dict(nosubclass_df.label,oversampling_limit),random_state=42)
            else:
                sm = SMOTE(random_state=42)
            X_smote, y_smote = sm.fit_sample(X,nosubclass_df.label)
            return X_smote, y_smote
        else:
            if mode.split('+')[1] == 'KNN':
                print('SMOTE+KNN')
                if undersampling_limit and undersampling_limit >= minimum_elements:
                    cc = ClusterCentroids(sampling_strategy=sampling_dict(nosubclass_df.label,undersampling_limit))
                else:
                    cc = ClusterCentroids()
                x, y = cc.fit_sample(X,nosubclass_df.label)
                sm = SMOTE(random_state=42)
                return sm.fit_sample(x,y)
            #TODO still have to figure out to tell Tomek algorithm I want a specific number of samples for each label
            #elif mode.split('+')[1] == 'Tomek':
                #print('SMOTE+Tomek')
                #if undersampling_limit and undersampling_limit >= minimum_elements:
                #    tl = TomekLinks(sampling_strategy=tomek_ratio)
                #else:
                #    tl = TomekLinks()
                #x , y = tl.fit_sample(X,nosubclass_df.label)
                #sm = SMOTE(random_state=42)
                #return sm.fit_sample(x,y)


X_centroids,y_centroids = downsample(mode='centroids')
X_smote_knn,y_smote_knn = downsample(mode='SMOTE+KNN',undersampling_limit=100,oversampling_limit=500)
#X_smote_tomek,y_smote_tomek = downsample(mode='SMOTE+Tomek',undersampling_limit=100,oversampling_limit=500)

#TODO for some reason i cannot get to color the data the way I want to
def PCA_plot(data,labels):
    pca = PCA(n_components=2)
    PCA_data = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = PCA_data, columns=['PC1','PC2'])
    print(data.shape)
    print(PCA_data.shape)
    #principalDf['target'] = labels
    #finalDf = principalDf
    
    import matplotlib.pyplot as plt

    plt.scatter(PCA_data[:,0], PCA_data[:,1],
            c=list(labels), edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

PCA_plot(X_centroids,y_centroids)


