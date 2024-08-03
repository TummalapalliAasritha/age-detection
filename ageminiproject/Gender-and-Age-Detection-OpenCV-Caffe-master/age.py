import cv2
import numpy as np
from collections import deque

# Mean values for the Caffe model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = [
    '(0, 5)', '(6, 10)', '(11, 15)', '(16, 20)', '(21, 25)', 
    '(26, 30)', '(31, 35)', '(36, 40)', '(41, 45)', '(46, 50)', 
    '(51, 55)', '(56, 60)', '(61, 65)', '(66, 70)', '(71, 75)', 
    '(76, 80)', '(81, 85)', '(86, 90)', '(91, 95)', '(96, 100)'
]

# Initialize the webcam
def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    return cap

def initialize_age_model():
    age_net = cv2.dnn.readNetFromCaffe(
        r'C:\Users\DELL\OneDrive\Desktop\ageminiproject\Gender-and-Age-Detection-OpenCV-Caffe-master\data\deploy_age.prototxt',
        r'C:\Users\DELL\OneDrive\Desktop\ageminiproject\Gender-and-Age-Detection-OpenCV-Caffe-master\data\age_net.caffemodel')
    return age_net

def detect_age(age_net, face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_index = age_preds[0].argmax()
    age = age_list[age_index]
    return age

def process_camera_feed(age_net, cap):
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier(r'C:\Users\DELL\OneDrive\Desktop\ageminiproject\Gender-and-Age-Detection-OpenCV-Caffe-master\data\haarcascade_frontalface_alt.xml')

    age_predictions = deque(maxlen=10)  # Store the last 10 predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            face_img = cv2.resize(face_img, (227, 227))  # Resize face image to match model input size

            # Predict Age
            age = detect_age(age_net, face_img)
            age_predictions.append(age)

            # Calculate the most common age prediction in the deque
            most_common_age = max(set(age_predictions), key=age_predictions.count)
            print(f"Age Range: {most_common_age}")

            # Overlay text
            overlay_text = f"{most_common_age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, overlay_text, (x, y-10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Age Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = initialize_camera()
    age_net = initialize_age_model()
    process_camera_feed(age_net, cap)
