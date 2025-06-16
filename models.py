import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, future, morphology
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial

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

#roiX = [
#roiY

training_imgOne = io.imread(pred_file)[0]

roi = io.imread(raw_file)[0][130:200,0:760]
mask = training_imgOne[130:200,0:760]
training_labels = mask.astype(np.uint8) + 1

test_img_features = features_func(roi)

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
    mlp_clf = MLPClassifier(hidden_layer_sizes=800,random_state=1, max_iter=300)
    mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)
    return mlp_clf

def forest_model():
    forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05)
    forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)
    return forest_clf
