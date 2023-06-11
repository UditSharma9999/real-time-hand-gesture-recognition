import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


# initialize mediapipe
mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, 
                      min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


cap=cv2.VideoCapture(0)
directory='Image/'
while True:
    _,frame=cap.read()
    count = {
                'Grab': len(os.listdir(directory+"/Grab")),
                'point': len(os.listdir(directory+"/point")),
                'Stop': len(os.listdir(directory+"/Stop")),
                'thumb': len(os.listdir(directory+"/thumb")),
             }
    
    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])

    frame=frame[40:400,0:300]
    # frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    # print(result)

    landmarks = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
    if landmarks:
        arr = []
        for landmark in landmarks.landmark:
            arr.append(landmark.x)
            arr.append(landmark.y)
            arr.append(landmark.z)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        with open("data.csv", 'a', newline="") as f:
            arr.insert(0, 0)
            writer = csv.writer(f)
            writer.writerow(arr)

    if interrupt & 0xFF == ord('b'):
        with open("data.csv", 'a', newline="") as f:
            arr.insert(0, 1)
            writer = csv.writer(f)
            writer.writerow(arr)

    if interrupt & 0xFF == ord('c'):
        with open("data.csv", 'a', newline="") as f:
            arr.insert(0, 2)
            writer = csv.writer(f)
            writer.writerow(arr)

    if interrupt & 0xFF == ord('d'):
        with open("data.csv", 'a', newline="") as f:
            arr.insert(0, 3)
            writer = csv.writer(f)
            writer.writerow(arr)

