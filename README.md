# cv_chest

# step 1 - script for reading images and normalize data (maybe)
# step 2 - divide data in category
- a) define categories (classes). We already have labels but for a convolutional neural network (CNN) , if we want to use it, classification works best if each data (image) appertains to only one class/has only one lables. 
Or maybe just reuse images. If an image has two or more labels just consider it as a member of two or more classes.
- b) if one image belongs to more than one category (class) we should try label it as best as we can (KNN?)
- c) consider data augmentation or downloading whole dataset.


interesting links on image procesing: 

https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

https://www.kdnuggets.com/2017/06/medical-image-analysis-deep-learning-part-3.html

Very nice article on Convolutional Neural NEtwork....I think this would be a nice thing to implement

https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d#:~:text=Convolutional%20Neural%20Networks%20(CNNs)%20is,used%20for%20image%20classification%20problem.&text=Instead%20of%20a%20fully%20connected,small%20patch%20of%20the%20image.

Keras with Tensor Flow for image processing

https://keras.io/getting_started/intro_to_keras_for_engineers/

https://keras.io/examples/vision/image_classification_from_scratch/
