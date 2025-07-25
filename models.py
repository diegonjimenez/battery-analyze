# Setting up modules
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, future, morphology
import pandas as pd
from sklearn import metrics
from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial
import tifffile as tiff
from tifffile import TiffFile
import os
import time #opt


# indices of real Truth, a folder of images where segmentation was manually done in order to aid training
real_indices = [1,17,65,145,217,322,392,413,444,530,580,668,1347,1403,1456,1508,1541,1731,1793,1827,1878,2397]
num_slices = 10
image_dir = "images/realTruths"

# number of images to test and number to train (can be edited)
num_test =  2
num_train = 10

# creates two arrays containing random indices of photos to train and test on
train_indices, test_indices =  train_test_split(np.array(range(3849)),train_size=.6,test_size=.4)

# removes the real indices from the test and split arrays
train_indices = np.setdiff1d(train_indices, real_indices)
test_indices = np.setdiff1d(test_indices, real_indices)

# Control vars for the model training (can be edited)
downsample = 2
sigma_min = 1
sigma_max = 20
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)


# Set up image paths for raw data and ground truth data. varies based off user
raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"

# An array containing the images to test on
imgRaw = []
imgPred = []
# An array containing the images to train on
rawImgStack = []
predImgStack = []

def convert_to_binary(image):
    # Takes in an image labeled 1 and 2s and turns the 1s to 0s
    # and turns the 2s to 1s
    # needed due to input images being in 1s and 2s instead of binary
    if np.isin(image, [1, 2]).all():
        return image-1
    edited = image
    for x in range(len(edited)):
        for y in range(len(edited[x])):
            if edited[x][y] == 2 or edited[x][y] == 255:
                edited[x][y] = 1
            elif edited[x][y] == 1:
                edited[x][y] = 0
    return edited

# Adds the manual segmentation to training
for i in range(1, num_slices + 1):
    mask_path = os.path.join(image_dir, f"grounded_mask_slice_{i} copy.tiff")
    raw_path = os.path.join(image_dir, f"grounded_raw_slice_{i} copy.tiff")

    mask_img = io.imread(mask_path)
    raw_img = io.imread(raw_path)

    imgRaw.append(raw_img)
    imgPred.append(convert_to_binary(mask_img))

    print(f"Loaded slice {i}: mask shape = {mask_img.shape}, raw shape = {raw_img.shape}")

# Appends the images to the training and testing arrays
with TiffFile(raw_file) as raw, TiffFile(pred_file) as pred:
    for x in range(num_test):
        imgRaw.append(raw.pages[test_indices[x]].asarray() )
        imgPred.append(pred.pages[test_indices[x]].asarray() )
    imgRaw = np.array(imgRaw)
    imgPred = np.array(imgPred)
    for x in range(num_train):
       rawImgStack.append(raw.pages[train_indices[x]].asarray()[::downsample,::downsample] )
       predImgStack.append(pred.pages[train_indices[x]].asarray()[::downsample,::downsample] )

# Stacks images to train the model off of one image
rawImgStack = np.vstack(rawImgStack)
predImgStack = np.vstack(predImgStack)

# Creates labels and features of the stacked image to train the models
training_labels = predImgStack.astype(np.uint8) + 1
test_img_features = features_func(rawImgStack)



#Set up models
def mlp_model():
    # Returns an mlp model trained on the battery ground truth
    start = time.time()
    mlp_clf = MLPClassifier(hidden_layer_sizes=400,random_state=1, max_iter=300)
    mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)
    end = time.time()
    print("Finished mlp forest, took ", end - start, " seconds")
    return mlp_clf

def forest_model():
    # Returns a random forest model trained on the battery ground truth
    start = time.time()
    forest_clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, max_depth=16, max_samples=0.02) #og n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05
    forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)
    end = time.time()
    print("Finished training forest, took ", end - start, " seconds")
    return forest_clf
