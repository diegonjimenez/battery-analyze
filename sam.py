# Adapted from Segment Anything (https://github.com/facebookresearch/segment-anything)
# Original License: Apache 2.0
# Modified by Diego Jimenez for [Your Project Name]
import os
import cv2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def find_dendrite_points(img,is_colored,counter = 100, amp=20):
    # This function uses a high threshold to find guaranteed dendrite points (pixels in the photo)
    # then marks every counter-th dendrite pixel as 1 and every counter*amp-th non dendrite marks as 0
    #rewrite later
    input_points = []
    input_labels = []
    dend_count = counter
    space_count = amp*counter

    if is_colored:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x][y]>238:
                dend_count += 1
                if dend_count >= counter:
                    input_points.append([y,x])
                    input_labels.append(1)
                    dend_count = 0
            elif (img[x][y] > 80) & (img[x][y]< 230):
                space_count += 1
                if space_count >= amp*counter:
                    input_points.append([y,x])
                    input_labels.append(0)
                    space_count = 0
    return [np.array(input_points), np.array(input_labels)]

# select the method for computation (CUDA, mps or, cpu powered)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

#Locate location of SAM checkpoint and data source will vary off user
sam2_checkpoint = "/Users/pureduck/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

#Instantiate the SAM model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def sam_predict(img):
    # Takes in a raw battery image
    # returns a SAM produced segmentation image
    # using the find_dendrite_points method
    if img[0].shape == 1:
        img = np.array(img.convert("RGB"))
    predictor.set_image(img)
    input_point, input_label = find_dendrite_points(img, True, 2, 50)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    return masks[2]
