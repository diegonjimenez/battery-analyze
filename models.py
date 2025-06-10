import numpy as np
from skimage import io, filters
from skimage import morphology, feature, future
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial

raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"

training_thresholded_img = io.imread(raw_file)[0] > filters.threshold_triangle(io.imread(raw_file)[0])

roi = io.imread(raw_file)[0][130:200,0:760]
mask = training_thresholded_img[130:200,0:760]
training_labels = mask.astype(np.uint8) + 1
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

test_img_features = features_func(roi)


def mlp_model():
    mlp_clf = MLPClassifier(hidden_layer_sizes=800,random_state=1, max_iter=300)
    mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)
    return mlp_clf

def forest_model():
    forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05)
    forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)
    return forest_clf
