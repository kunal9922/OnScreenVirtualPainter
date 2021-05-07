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

# Getting the HandDetector for handDetection
detector = htm.HandDetector(detectionCon=0.85)

#read from WebCam here
cap = cv2.VideoCapture(r"dataSets//handTestVideo.mp4")
while cap.isOpened():
      #1. Import Image
      success, imgFrame = cap.read()
      imgFrame = cv2.resize(imgFrame, (720, 480))
      imgFrame = cv2.flip(imgFrame, 1)

      #2. FindHand LandMarks
      imgFrame = detector.findHands(imgFrame)
      lmList = detector.findPosition(imgFrame, draw=False)

      # check if Hand is detected or not if yes then process farther
      if len(lmList) != 0:
            fingers = [] # this list tell us which finger is up and which finger is down
            
            #tip of index and middle fingers
            x1, y1 = lmList[8][1:]  #  Index finger coordinates
            x2, y2 = lmList[12][1:]
            
            #3. Check which fingers are up
            fingers = detector.fingersUp()
            print(fingers)
                  
            #4. If selection mode - Two fingers are up
            #5. If Drawing Mode - Index finger is Up

      
      
      #Setting the header image
      imgFrame[0:130, 0:720] = header1
      cv2.imshow("TestFrames", imgFrame)
      
      cv2.waitKey(1)
