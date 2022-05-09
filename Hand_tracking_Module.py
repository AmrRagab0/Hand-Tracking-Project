import cv2 
import mediapipe as mp

import time


class handDetector():
	def __init__(self,mode=False,maxHands=2,detectionCon = 0.5,trackCon = 0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands

		self.hands = self.mpHands.Hands()# check parameters

		self.mpDraw = mp.solutions.drawing_utils

	def findHands(self,img,draw = True):
		imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		self.result = self.hands.process(imgRGB) # hands module only works on RGB
		if (self.result.multi_hand_landmarks):
			for hand in self.result.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img,hand,self.mpHands.HAND_CONNECTIONS)  # draws the lines of fingers

				
		return img		

	def findPosition(self,img,handNo =0,draw =True):
		lmlist = []   # landmark list
		if (self.result.multi_hand_landmarks):
			myHand = self.result.multi_hand_landmarks[handNo]
			for ID,lm in enumerate(myHand.landmark):
				h,w,c = img.shape
				cx,cy = int(lm.x*w),int(lm.y*h)  # the position of landmark
				lmlist.append(cx,cy)
				if draw :
					cv2.circle(img,(cx,cy),14,(255,0,255),cv2.FILLED)


		return lmlist
#while True: 
#	success , img = cap.read()
	
	 

#	cv2.imshow("image",img)
#if cv2.waitKey(1) & 0xFF == ord('q'):
 ##         break




def main():
	ptime= 0
	curr_time= 0
	cap = cv2.VideoCapture(0)
	detector = handDetector()
	while True: 
		success , img = cap.read()
		img = detector.findHands(img)
		#lmlist = detector.findPosition(img)

		curr_time = time.time()
		#ptime = curr_time
		fps = 1/(curr_time - ptime)
		ptime = curr_time
	
		cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
		cv2.imshow("image",img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
	          break

if __name__== "__main__":
	main()
