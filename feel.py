"""
    Import necessary libraries:
    The code imports OpenCV (cv2), TensorFlow (tf), and Numpy (np) libraries to use their functions for image processing and deep learning.

    Load the pre-trained model:
    The code loads a pre-trained deep learning model ('emotion_detection_model.h5') using the TensorFlow Keras API.

    Load the face detection classifier:
    The code loads the pre-trained Haar Cascade classifier ('haarcascade_frontalface_default.xml') using OpenCV. The classifier is used to detect faces in an image.

    Start capturing frames from the webcam:
    The code initializes the camera using OpenCV's VideoCapture() function and captures frames continuously.

    Convert the captured frame to grayscale:
    The code converts the captured frame to grayscale using OpenCV's cvtColor() function.

    Detect faces in the grayscale image:
    The code uses the face detection classifier to detect faces in the grayscale image using OpenCV's detectMultiScale() function.

    Process each face detected:
    For each face detected, the code extracts the region of interest (ROI) from the grayscale image and resizes it to 48x48 pixels. It also normalizes the pixel values and adds two additional dimensions to the tensor to match the input shape of the pre-trained model.

    Make predictions using the pre-trained model:
    The code uses the pre-trained model to make predictions on the processed image of the face.

    Display the predicted emotion:
    The code displays the predicted emotion as text on the frame using OpenCV's putText() function.

    Display the resulting image:
    The code displays the resulting image in a window using OpenCV's imshow() function.

    Quit the program:
    The code waits for a keyboard event to exit the program using OpenCV's waitKey() function.
"""

import cv2
import tensorflow as tf
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/kali/data/haarcascades/haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('/home/kali/Downloads/emotion_detection_model.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        prediction = model.predict(roi_gray)[0]
        label = emotion_labels[prediction.argmax()]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (x, y-10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
