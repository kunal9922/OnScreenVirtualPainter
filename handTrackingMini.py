import cv2
import mediapipe as mp
import time

class HandDetector():
      def __init__(self, mode=False, maxHands=2,detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.detectionCon = detectionCon
            self.trackCon = trackCon

            #import the class  from mediapipe
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                                                                          self.detectionCon, self.trackCon)
            self.mpDraw = mp.solutions.drawing_utils # -----this method allows us to drawlines on hands
            self.tipIds = [4, 8, 12, 16, 20] # ids for every finger in our hands
            
      def findHands(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)  #           result contains the detected landmarks as dictionary 
            # print(results.multi_hand_landmarks)

            # --- detected the hand 
            if self.results.multi_hand_landmarks:
                  # get the how many hand we detected in an img.
                  for handlms in self.results.multi_hand_landmarks:
                           if draw:
                                 self.mpDraw.draw_landmarks(img, handlms,
                                                                                                      self.mpHands.HAND_CONNECTIONS)
            return img

      def findPosition(self, img, handNo=0, draw=True):
            self.lmList = [] # will store landmarks
             # --- detected the hand 
            if self.results.multi_hand_landmarks:
                  myHand = self.results.multi_hand_landmarks[handNo]
                  for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.lmList.append([id, cx, cy])
                        

            return self.lmList

      def fingersUp(self):
            fingers = []  # this list tell us which finger is up and which finger is down
      
            # Thumb
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                  fingers.append(1)
            else:
                  fingers.append(0)

            # 4 Fingers
            for id in range(1,5):
                  if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                        fingers.append(1)
                  else:
                        fingers.append(0)
                        
            return fingers

def workWithHandsDetect():
      pTime = 0
      cTime = 0
      # create a detector for detect the hands landmarks
      handsDetector = HandDetector()
      
      cap = cv2.VideoCapture(r"dataSet\handTestVideo.mp4")
      while cap.isOpened():
            success, img = cap.read()
            img = cv2.resize(img, (720,480))
            img = handsDetector.findHands(img)
            #lmList = handsDetector.findPosition(img)
           # if len(lmList) != 0:
                #  print(lmList[3])
                #pass
            #calculate the frame rate framePerSecond fps.
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
            cv2.imshow("Hand", img)
            cv2.waitKey(1)
            
workWithHandsDetect()
