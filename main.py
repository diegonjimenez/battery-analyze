from models import *
from sam import *

#instantiating trained models from models.py
forest_model = forest_model()
mlp_model = mlp_model()


# Produces the segmentation results
forest_results = []
mlp_results = []
threshold_results = []
median_results = []
sam_results = []


for x in range(num_test):
    img_feature = features_func(imgRaw[x])
    median_features = features_func(filters.median(imgRaw[x] , morphology.footprint_rectangle((3,3))))

    forest_results.append(convert_to_binary( future.predict_segmenter(img_feature,forest_model)) )
    mlp_results.append(convert_to_binary( future.predict_segmenter(img_feature,mlp_model)) )
    threshold_results.append(imgRaw[x] > 235)
    median_results.append(convert_to_binary(future.predict_segmenter(median_features, forest_model)))
    sam_results.append(sam_predict( cv2.cvtColor(imgRaw[x], cv2.COLOR_GRAY2RGB)  ))

# The following produces a plot for a comparison of one segmentation result
f, ax = plt.subplots(3, 2, figsize=(16, 9))
ax[0][0].imshow(imgRaw[0], cmap='cividis')
ax[0][0].set_title("Raw Image")

ax[0][1].imshow(imgPred[0],cmap = "cividis")
ax[0][1].set_title("'Ground Truth' Image")

ax[1][0].imshow(threshold_results[0],cmap='cividis')
ax[1][0].set_title("Thresholded Image")

ax[1][1].imshow(forest_results[0], cmap = "cividis")
ax[1][1].set_title("Random Forest Image")

ax[2][0].imshow(mlp_results[0],cmap='cividis')
ax[2][0].set_title("MLP Image")

ax[2][1].imshow(sam_results[0], cmap = "cividis")
ax[2][1].set_title("Sam")

f.suptitle("Comparisons of One Image")

#Determining metrics for each method

precisions = {"threshold": [], "forest": [], "mlp":[], "median":[], "SAM":[]}
ious = {"threshold": [], "forest": [], "mlp":[], "median":[], "SAM":[]}
f1s = {"threshold": [], "forest": [], "mlp":[], "median":[], "SAM":[]}


for x in range(num_test):
    precisions["threshold"].append( metrics.accuracy_score(imgPred[x].flatten(), threshold_results[x].flatten()) )
    precisions["forest"].append( metrics.accuracy_score(imgPred[x].flatten(), forest_results[x].flatten()) )
    precisions["mlp"].append( metrics.accuracy_score(imgPred[x].flatten(), mlp_results[x].flatten()) )
    precisions["median"].append( metrics.accuracy_score(imgPred[x].flatten(), median_results[x].flatten()) )
    precisions["SAM"].append(metrics.accuracy_score(imgPred[x].flatten(), sam_results[x].flatten()))

    ious["threshold"].append(metrics.jaccard_score(imgPred[x].flatten(), threshold_results[x].flatten(), average="binary"))
    ious["forest"].append(metrics.jaccard_score(imgPred[x].flatten(), forest_results[x].flatten(), average="binary"))
    ious["mlp"].append(metrics.jaccard_score(imgPred[x].flatten(), mlp_results[x].flatten(), average="binary"))
    ious["median"].append(metrics.jaccard_score(imgPred[x].flatten(), median_results[x].flatten(), average="binary"))
    ious["SAM"].append(metrics.jaccard_score(imgPred[x].flatten(), sam_results[x].flatten(), average="binary"))

    f1s["threshold"].append(metrics.f1_score(imgPred[x].flatten(), threshold_results[x].flatten(), average="binary"))
    f1s["forest"].append(metrics.f1_score(imgPred[x].flatten(), forest_results[x].flatten(), average="binary"))
    f1s["mlp"].append(metrics.f1_score(imgPred[x].flatten(), mlp_results[x].flatten(), average="binary"))
    f1s["median"].append(metrics.f1_score(imgPred[x].flatten(), median_results[x].flatten(), average="binary"))
    f1s["SAM"].append(metrics.f1_score(imgPred[x].flatten(), sam_results[x].flatten(), average="binary"))

threshold_precision = f" {round( np.average(precisions["threshold"]) ,3 )} ({round(np.std(precisions["threshold"]), 3) } sd)"
forest_precision = f" {round(np.average(precisions["forest"]),3)} ({round(np.std(precisions["forest"]), 3)} sd)"
mlp_precision = f"{round(np.average(precisions["mlp"]), 3)} ({round(np.std(precisions["mlp"]), 3)} sd)"
median_precision = f"{round(np.average(precisions["median"]), 3)} ({round(np.std(precisions["median"]), 3)} sd)"
SAM_precision = f"{round(np.average(precisions["SAM"]), 3)} ({round(np.std(precisions["SAM"]), 3)} sd)"

threshold_iou = f"{round(np.average(ious["threshold"]), 3)} ({round(np.std(ious["threshold"]), 3)} sd)"
forest_iou = f"{round(np.average(ious["forest"]), 3)} ({round(np.std(ious["forest"]), 3)} sd)"
mlp_iou = f"{round(np.average(ious["mlp"]), 3)} ({round(np.std(ious["mlp"]), 3)} sd)"
median_iou = f"{round(np.average(ious["median"]), 3)} ({round(np.std(ious["median"]), 3)} sd)"
SAM_iou = f"{round(np.average(ious["SAM"]), 3)} ({round(np.std(ious["SAM"]), 3)} sd)"

threshold_f1 = f"{round(np.average(f1s["threshold"]), 3)} ({round(np.std(f1s["threshold"]), 3)} sd)"
forest_f1 = f"{round(np.average(f1s["forest"]), 3) } ({round(np.std(f1s["forest"]), 3)} sd)"
mlp_f1 = f"{round(np.average(f1s["mlp"]), 3)} ({round(np.std(f1s["mlp"]), 3)} sd)"
median_f1 = f"{round(np.average(f1s["median"]), 3)} ({round(np.std(f1s["median"]), 3)} sd)"
SAM_f1 = f"{round(np.average(f1s["SAM"]), 3)} ({round(np.std(f1s["SAM"]), 3)} sd)"

metrics = {"Thresholding":[threshold_precision,threshold_iou,threshold_f1], "Forest":[forest_precision,forest_iou,forest_f1],"MLP":[mlp_precision,mlp_iou,mlp_f1],"Median Forest":[median_precision,median_iou,median_f1],"SAM":[SAM_precision,SAM_iou,SAM_f1]}


if num_test >1:
    score_names = ["mPrecision", "mIoU", "mF1/mDice"]
else:
    score_names = ["Precision","IoU","F1/Dice"]

df = pd.DataFrame(metrics,index=score_names)
df.to_csv('/Users/pureduck/Downloads/output.csv', index=True)
print(df)

plt.show()
