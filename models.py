#SEC 1: Setting up modules
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

#SEC 2: Setting up control vars

#whether to display comparisons between one image or not
classic = True
num_test = 5

train_indices, test_indices =  train_test_split(np.array(range(3849)),train_size=.7,test_size=.30)
train_indices = train_indices[:2]

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


# SEC 3: Setting up images
raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"
imgRaw = []
imgPred = []
rawImgStack = []
predImgStack = []

#yml

with TiffFile(raw_file) as raw, TiffFile(pred_file) as pred:
    if classic:
        imgRaw = raw.pages[test_indices[0]].asarray()
        imgPred = pred.pages[test_indices[0]].asarray()
    else:
        for x in range(num_test):
            imgRaw.append(raw.pages[test_indices[x]].asarray() )
            imgPred.append(pred.pages[test_indices[x]].asarray() )
        imgRaw = np.array(imgRaw)
        imgPred = np.array(imgPred)
        plt.imshow(imgRaw[0])
        plt.show()
    for x in train_indices:
       #plt.imshow(raw.pages[trainNum[x]].asarray()[::downsample,::downsample])
       #plt.show()
       rawImgStack.append(raw.pages[x].asarray()[::downsample,::downsample] )
       predImgStack.append(pred.pages[x].asarray()[::downsample,::downsample] )

predImgStack = np.vstack(predImgStack)
rawImgStack = np.vstack(rawImgStack)


training_labels = predImgStack.astype(np.uint8) + 1
test_img_features = features_func(rawImgStack)

print("Training_labels shape is ", training_labels.shape)
print("test_img_features shape is ", test_img_features.shape)

#SEC 3: Models!

def convert_to_binary(image):
    if np.isin(image, [1, 2]).all():
        return image-1
    edited = image
    for x in range(len(edited)):
        for y in range(len(edited[x])):
            if edited[x][y] == 2:
                edited[x][y] = 1
            elif edited[x][y] == 1:
                edited[x][y] = 0
    return edited

def mlp_model():
    mlp_clf = MLPClassifier(hidden_layer_sizes=400,random_state=1, max_iter=300)
    mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)
    print("Finished mlp forest")
    return mlp_clf

def forest_model():
    forest_clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, max_depth=16, max_samples=0.02) #og n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05
    forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)
    print("Finished training forest")
    return forest_clf
