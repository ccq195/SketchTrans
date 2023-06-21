import numpy as np
import argparse
import time
import cv2
import os
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default=os.path.sep.join(['/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/test/10_03_22011.jpg']),
                help="path to input image that we'll apply GrabCut to")
ap.add_argument("-mask", "--mask", type=str,
                default=os.path.sep.join(['/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/test/t.jpeg']),
                help="path to input mask")
ap.add_argument("-c", "--iter", type=int, default=10,
                help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
image = imutils.resize(image, width=400)
mask = cv2.imread(args["mask"], cv2.IMREAD_GRAYSCALE)
print(image.shape, mask.shape)


roughOutput = cv2.bitwise_and(image, image, mask=mask)


cv2.imshow("Rough Output", roughOutput)
cv2.waitKey(0)


mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD


fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")


start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
                                       fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_MASK)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))


values = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
)


for (name, value) in values:
    print("[INFO] showing mask for '{}'".format(name))
    valueMask = (mask == value).astype("uint8") * 255


    cv2.imshow(name, valueMask)
    cv2.waitKey(0)


outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")


output = cv2.bitwise_and(image, image, mask=outputMask)


cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)
