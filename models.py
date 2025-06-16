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

#[250:340,0:2071]
training_labels = []
test_img_features = []
trainNum = np.array([0,0,751,980,2970])
#trainNum = np.array([0,0,375,550,751,980])
#roiX = np.array([[130,200],[250,340],[120,200],[50,110],[250,340],[30,130]])
#roiY = np.array([[0,760],[0,760],[1670,2071],[380,480],[0,2071],[0,2071]])
roiX = np.array([[130,200],[250,340],[250,340],[0,340],[0,340]])
roiY = np.array([[0,760],[0,760],[0,2071],[0,2071],[0,2071]])

for x in range(len(trainNum)):
   training_img = io.imread(pred_file)[trainNum[x]]
   roi = io.imread(raw_file)[trainNum[x] ][roiX[x][0]:roiX[x][1],roiY[x][0]:roiY[x][1]]
   mask = training_img[roiX[x][0]:roiX[x][1],roiY[x][0]:roiY[x][1]]
   training_labels.append( mask.astype(np.uint8) + 1 )
   test_img_features.append( features_func(roi) )
   #plt.imshow(mask)
   #plt.show()


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
    for x in range(len(training_labels)):
        mlp_clf = future.fit_segmenter(training_labels[x], test_img_features[x], mlp_clf)
    return mlp_clf

def forest_model():
    forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_samples=0.05,warm_start=True)
    for x in range(len(training_labels)):
        forest_clf.n_estimators = forest_clf.n_estimators + 100
        forest_clf = future.fit_segmenter(training_labels[x], test_img_features[x], forest_clf)
    return forest_clf
