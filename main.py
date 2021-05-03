import numpy as np
import time
import os
import cv2
import handTrackingMini as htm

folderPath = "HeadersAIPainter"
myList = os.listdir(folderPath)
print(myList)

headerImgs = []
for imgPath in myList:
      image = cv2.imread(f'{folderPath}/{imgPath}')
      headerImgs.append(image)

print(len(headerImgs))
