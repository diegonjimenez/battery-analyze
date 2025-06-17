import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, future, morphology
from sklearn import metrics
from skimage.transform import rescale
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial

#SEC 1
raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"
sigma_min = 1
sigma_max = 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)

#[250:340,0:2071]

rawImgStack = []
predImgStack = []

trainNum = np.array([0,751,980,1900,2970])

for x in range(len(trainNum)):
   rawImgStack.append(io.imread(raw_file)[trainNum[x]])
   predImgStack.append(io.imread(pred_file)[trainNum[x]])

predImgStack = np.vstack(predImgStack)
rawImgStack = np.vstack(rawImgStack)

training_labels = predImgStack.astype(np.uint8) + 1
test_img_features = features_func(rawImgStack)


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
    mlp_clf = MLPClassifier(hidden_layer_sizes=800,random_state=1, max_iter=300,warm_start=True)
    mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)
    return mlp_clf

def forest_model():
    forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05,warm_start=True)
    forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)
    return forest_clf
