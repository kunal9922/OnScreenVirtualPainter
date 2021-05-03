import numpy as np
import time
import os
import cv2
import handTrackingMini as htm

folderPath = "HeadersAIPainter"
myList = os.listdir(folderPath)
print(myList)

# read the header images for paint tools selection
headerImgs = []
for imgPath in myList:
      image = cv2.imread(f'{folderPath}/{imgPath}')
      headerImgs.append(image)

print(len(headerImgs))

#deafault header image for nothing to selected
header1 = headerImgs[2]
header1 = cv2.resize(header1, (720, 130))

#read from WebCam here
cap = cv2.VideoCapture(r"dataSets//handTestVideo.mp4")
while cap.isOpened():
      success, imgFrame = cap.read()
      imgFrame = cv2.resize(imgFrame, (720, 480))

      #Setting the header image
      imgFrame[0:130, 0:720] = header1
      cv2.imshow("TestFrames", imgFrame)
      
      cv2.waitKey(1)
