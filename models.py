#SEC 1: Setting up modules
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, future, morphology
import pandas as pd
from sklearn import metrics
from skimage.transform import rescale
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial
import tifffile as tiff
from tifffile import TiffFile

#SEC 2: Setting up control vars

future_index = 121#np.random.randint(0, 3849)

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
rawImgStack = []
predImgStack = []

trainNum = np.array([0,751,980,2970]) #

with TiffFile(raw_file) as raw, TiffFile(pred_file) as pred:
    imgRaw = raw.pages[future_index].asarray()
    imgPred = pred.pages[future_index].asarray()
    for x in range(len(trainNum)):
       rawImgStack.append(raw.pages[trainNum[x]].asarray()[::downsample,::downsample] )
       predImgStack.append(pred.pages[trainNum[x]].asarray()[::downsample,::downsample] )

predImgStack = np.vstack(predImgStack)
rawImgStack = np.vstack(rawImgStack)


training_labels = predImgStack.astype(np.uint8) + 1
test_img_features = features_func(rawImgStack)

print("Training_labels shape is ", training_labels.shape)
print("test_img_features shape is ", test_img_features.shape)

#SEC 3: Models!

print("Passed Section 1") #OPTIONAL

# SEC 2
def convert_to_binary(image):
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
print("Passed Section 2")
