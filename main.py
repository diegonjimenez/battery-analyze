import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial


## sets up file and renders only first one
index = 100

raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"
imgRaw = io.imread(raw_file)[index]
imgPred = io.imread(pred_file)[index]

training_thresholded_img = io.imread(raw_file)[0] > filters.threshold_triangle(io.imread(raw_file)[0])
##

## Random forest setup

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

main_img_features = features_func(imgRaw)
test_img_features = features_func(roi)

#declaring and training model
forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05)
forest_clf = future.fit_segmenter(training_labels, test_img_features, forest_clf)

main_forest_result = future.predict_segmenter(main_img_features, forest_clf)

##

## MLP Classifier

mlp_clf = MLPClassifier(random_state=1, max_iter=300)
mlp_clf = future.fit_segmenter(training_labels, test_img_features, mlp_clf)

main_mlp_result = future.predict_segmenter(main_img_features, mlp_clf)

#median 3x3

##

f, ax = plt.subplots(3, 2, figsize=(16, 9))
ax[0][0].imshow(imgRaw, cmap='gray')
ax[0][0].set_title("Raw Image")

ax[0][1].imshow(imgPred,cmap = "gray")
ax[0][1].set_title("Predicted Image")

ax[1][0].imshow(imgRaw > filters.threshold_triangle(imgRaw),cmap='gray')
ax[1][0].set_title("Thresholded Image")

ax[1][1].imshow(main_forest_result, cmap = "gray")
ax[1][1].set_title("Random Forest Image")

ax[2][0].imshow(main_mlp_result,cmap='gray')
ax[2][0].set_title("MLP Image")

plt.show()

