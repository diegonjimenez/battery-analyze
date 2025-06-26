from models import *
import time #OPTIONAL


start_time = time.time() #OPTIONAL


## sets up file and renders only first one
index = future_index#np.random.randint(0, 3849)
medianImg = filters.median(imgRaw, morphology.footprint_rectangle((3,3)))

# Sets up features for training
main_img_features = features_func(imgRaw)
median_img_features = features_func(medianImg)

#Gets models and gets results for the diff methods
forest_model = forest_model()
mlp_model = mlp_model()
main_forest_result = future.predict_segmenter(main_img_features, forest_model)
main_mlp_result = future.predict_segmenter(main_img_features, mlp_model)
main_threshold_result = imgRaw > filters.threshold_otsu(imgRaw)
main_median_result = future.predict_segmenter(median_img_features, forest_model)

#converting the segmentations to binary for better accuracy on the metrics
main_forest_result = convert_to_binary(main_forest_result)
main_mlp_result = convert_to_binary(main_mlp_result)
main_median_result = convert_to_binary(main_median_result)

#median 3x3 rec for median filter - dani

# Setting Graph

print("Index is ", index) #OPTIONAL

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
ax[2][1].set_title("Forest w/ Gaussian before")

#Determining metrics for each method

threshold_precision = metrics.accuracy_score(imgPred.flatten(), main_threshold_result.flatten())
forest_precision = metrics.accuracy_score(imgPred.flatten(), main_forest_result.flatten())
mlp_precision = metrics.accuracy_score(imgPred.flatten(), main_mlp_result.flatten())
median_precision = metrics.accuracy_score(imgPred.flatten(), main_median_result.flatten())


threshold_iou = metrics.jaccard_score(imgPred.flatten(), main_threshold_result.flatten(),average="binary")
forest_iou = metrics.jaccard_score(imgPred.flatten(), main_forest_result.flatten(),average="binary")
mlp_iou = metrics.jaccard_score(imgPred.flatten(), main_mlp_result.flatten(),average="binary")
median_iou = metrics.jaccard_score(imgPred.flatten(), main_median_result.flatten(),average="binary")

threshold_f1 = metrics.f1_score(imgPred.flatten(), main_threshold_result.flatten(),average="binary")
forest_f1 = metrics.f1_score(imgPred.flatten(), main_forest_result.flatten(),average="binary")
mlp_f1 = metrics.f1_score(imgPred.flatten(), main_mlp_result.flatten(),average="binary")
median_f1 = metrics.f1_score(imgPred.flatten(), main_median_result.flatten(),average="binary")


metrics = {"Thresholding":[threshold_precision,threshold_iou,threshold_f1], "Forest":[forest_precision,forest_iou,forest_f1],"MLP":[mlp_precision,mlp_iou,mlp_f1],"Median Forest":[median_precision,median_iou,median_f1]}

df = pd.DataFrame(metrics,index=["Precision","IoU","F1/Dice"])
print(df)

end_time = time.time() #OPTIONAL
print("It took ", end_time - start_time, " seconds") #OPTIONAL

plt.show()

