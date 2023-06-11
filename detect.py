import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("./model")
classNames = ["grab", "point", "stop", "thumb"]

cap=cv2.VideoCapture(0)
directory='Image/'
while True:
    _,frame=cap.read()
    x, y, c = frame.shape

    className = ""

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    arr = []
    landmarks = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
    if landmarks:
        for landmark in landmarks.landmark:
            arr.append(landmark.x)
            arr.append(landmark.y)
            arr.append(landmark.z)

    if len(arr)>0:
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)


        pred = model.predict([arr])
        classID = np.argmax(pred)
        className = classNames[classID]
        print(className)

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)
    
    cv2.imshow("Output", frame) 

    
    if cv2.waitKey(1) == ord('q'):
        break