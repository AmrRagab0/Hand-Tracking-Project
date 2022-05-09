import cv2 
import mediapipe as mp

import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands(False)    # check parameters

mpDraw = mp.solutions.drawing_utils
ptime= time.time()

while True: 
	success , img = cap.read()
	imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	result = hands.process(imgRGB) # hands module only works on RGB
	if (result.multi_hand_landmarks):
		for hand in result.multi_hand_landmarks:
			for ID,lm in enumerate(hand.landmark):
				h,w,c = img.shape
				cx,cy = int(lm.x*w),int(lm.y*h)  # the position of landmark
				cv2.circle(img,(cx,cy),14,(255,0,255),cv2.FILLED)
			mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)  # draws the lines of fingers

	curr_time = time.time()
	#ptime = curr_time
	fps = 1/(curr_time - ptime)
	ptime = curr_time
	
	cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


	cv2.imshow("image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
          break

