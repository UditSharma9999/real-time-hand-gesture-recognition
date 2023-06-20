# hand-gesture-recognition-using-mediapipe
Estimate hand pose using MediaPipe (Python version).<br> This is a sample 
program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.


https://github.com/UditSharma9999/real-time-hand-gesture-recognition/assets/63443176/706df366-72c3-4627-95f7-c9fe70d1e3c9


 **WebSite Link:** https://gamma.app/docs/Hand-Pose-Detection-Enhancing-HumanComputer-Interaction-yrdg61g80p9vbq5?mode=doc

This repository contains the following contents.
* Sample program
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later 
* matplotlib 3.3.2 or Later 

# Demo
Here's how to run the demo using your webcam.
```bash
python detect.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

### detect.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### code.ipynb
This is a model training script for hand sign recognition.
This is a model training script for finger gesture recognition.


# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.


# Reference
* [MediaPipe](https://mediapipe.dev/)

# Author
Udit Sharma
