import matplotlib.pyplot as plt
from models import *


## sets up file and renders only first one
index = np.random.randint(0, 3849)

raw_file = "images/8bitRawXCT.tif"
pred_file = "images/binaryPredicted.tif"
imgRaw = io.imread(raw_file)[index]
imgPred = io.imread(pred_file)[index]

main_img_features = features_func(imgRaw)
median_img_features = features_func(filters.median(imgRaw, morphology.footprint_rectangle((5,5))))

forest_model = forest_model()
mlp_model = mlp_model()

main_forest_result = future.predict_segmenter(main_img_features, forest_model)
main_mlp_result = future.predict_segmenter(main_img_features, mlp_model)
median_forest_result = future.predict_segmenter(median_img_features, forest_model)

#median 3x3

# Setting Graph

print(index)

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

ax[2][1].imshow(median_forest_result, cmap = "gray")
ax[2][1].set_title("Forest w/ Median Filter")

plt.show()

