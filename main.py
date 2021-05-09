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
drawColor = (189,28,95)
for imgPath in myList:
      image = cv2.imread(f'{folderPath}/{imgPath}')
      headerImgs.append(image)

print(len(headerImgs))

# Initials drawing points
x0, y0 = 0, 0
brushThickness = 5
eraserThickness = 10

# firstly draw on canvas
canvas = np.zeros((720, 960), dtype=np.uint8)
#deafault header image for selected purple
header = headerImgs[0]
header = cv2.resize(header, (720, 130))

# Getting the HandDetector for handDetection
detector = htm.HandDetector(detectionCon=0.85)

#read from WebCam here
cap = cv2.VideoCapture("dataSets//testDrawAndSelect.mp4")
print(cap.isOpened())
while cap.isOpened():
      #1. Import Image
      success, imgFrame = cap.read()
      imgFrame = cv2.resize(imgFrame, (720, 960))
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
           
                  
            #4. If selection mode - Two fingers are up
            if fingers[1] and fingers[2]:
                  print("Selection Mode")
                 
                  #check if our finger on header area(Clicking checking)
                  if y1 < 130:
                        if 100< x1 < 250: # purple selected
                              header =  cv2.resize(headerImgs[0], (720, 130))
                              drawColor = (189,28,95) # set purple color
                        elif 250 < x1 < 420: # pink selected
                              header = cv2.resize(headerImgs[1], (720, 130))
                              drawColor = (186, 18, 201) # set pink color
                        elif 420 < x1 < 550: # green selected
                              header = cv2.resize(headerImgs[2], (720, 130))
                              drawColor  = (0, 255, 0) # set green color
                        elif 550 < x1 < 720: # rubber selected
                              header = cv2.resize(headerImgs[3], (720, 130))
                              drawColor = (0, 0, 0)
                              
                        cv2.rectangle(imgFrame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED) 
                        
                        
                  
            #5. If Drawing Mode - Index finger is Up
            if fingers[1] and fingers[2] == False:
                  cv2.circle(imgFrame, (x1, y1), 15, drawColor, cv2.FILLED)
                  print("Drawing mode")
                  # checking if draw is just initate now
                  if x0 == 0 and y0 == 0:
                        x0, y0 = x1 , y1

                  if drawColor == (0,0,0): # Erase the color which had been drawed
                        cv2.line(imgFrame, (x0, y0), (x1, y1), drawColor, eraserThickness)
                        cv2.line(canvas, (x0, y0), (x1, y1), drawColor, eraserThickness)

                  else:
                        #drawing Go!
                        cv2.line(imgFrame, (x0, y0), (x1, y1), drawColor, brushThickness)
                        cv2.line(canvas, (x0, y0), (x1, y1), drawColor, brushThickness)
                  x0, y0 = x1, y1 # set current state to previous state
                  
                  
      
      #Setting the header image
      imgFrame[0:130, 0:720] = header
      cv2.imshow("TestFrames", imgFrame)
      cv2.imshow("Canvas", canvas)
      
      cv2.waitKey(1)
