from models import *
import time #OPTIONAL
import pandas as pd

start_time = time.time() #OPTIONAL

## sets up file and renders only first one
index = 1234#np.random.randint(0, 3849)
imgRaw = io.imread(raw_file)[index]
imgPred = io.imread(pred_file)[index]

main_img_features = features_func(imgRaw)
median_img_features = features_func(filters.median(imgRaw, morphology.footprint_rectangle((5,5))))

forest_model = forest_model()
mlp_model = mlp_model()

main_forest_result = future.predict_segmenter(main_img_features, forest_model)
main_mlp_result = future.predict_segmenter(main_img_features, mlp_model)
main_threshold_result = imgRaw > filters.threshold_triangle(imgRaw)
main_median_result = future.predict_segmenter(median_img_features, forest_model)

#converting the segmentations to binary for better accuracy on the metrics
main_forest_result = convert_to_binary(main_forest_result)
main_mlp_result = convert_to_binary(main_mlp_result)
main_median_result = convert_to_binary(main_median_result)

#median 3x3 rec for median filter - dani

# Setting Graph
#python pillow for imgs

print(index)

f, ax = plt.subplots(3, 2, figsize=(16, 9))
ax[0][0].imshow(imgRaw, cmap='gray')
ax[0][0].set_title("Raw Image")

ax[0][1].imshow(imgPred,cmap = "gray")
ax[0][1].set_title("'Ground Truth' Image")

ax[1][0].imshow(main_threshold_result,cmap='gray')
ax[1][0].set_title("Thresholded Image")

ax[1][1].imshow(main_forest_result, cmap = "gray")
ax[1][1].set_title("Random Forest Image")

ax[2][0].imshow(main_mlp_result,cmap='gray')
ax[2][0].set_title("MLP Image")

ax[2][1].imshow(main_median_result, cmap = "gray")
ax[2][1].set_title("Forest w/ Median Filter")

#Determining metrics for each method

threshold_iou = metrics.jaccard_score(imgPred.flatten(), main_threshold_result.flatten(),average="binary")
forest_iou = metrics.jaccard_score(imgPred.flatten(), main_forest_result.flatten(),average="binary")
mlp_iou = metrics.jaccard_score(imgPred.flatten(), main_mlp_result.flatten(),average="binary")
median_iou = metrics.jaccard_score(imgPred.flatten(), main_median_result.flatten(),average="binary")

threshold_f1 = metrics.f1_score(imgPred.flatten(), main_threshold_result.flatten(),average="binary")
forest_f1 = metrics.f1_score(imgPred.flatten(), main_forest_result.flatten(),average="binary")
mlp_f1 = metrics.f1_score(imgPred.flatten(), main_mlp_result.flatten(),average="binary")
median_f1 = metrics.f1_score(imgPred.flatten(), main_median_result.flatten(),average="binary")


metrics = {"Thresholding":[threshold_iou,threshold_f1], "Forest":[forest_iou,forest_f1],"MLP":[mlp_iou,mlp_f1],"Median Forest":[median_iou,median_f1]}

df = pd.DataFrame(metrics,index=["IoU","F1/Dice"])
print(df)

end_time = time.time() #OPTIONAL
print("It took ", end_time - start_time, " seconds") #OPTIONAL

plt.show()

