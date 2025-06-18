import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained CNN model
CNNmodel = load_model("my_model.keras")  # Ensure your model file exists
print(CNNmodel.input_shape)


# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face ROI from the original color frame
        face_roi = frame[y:y + h, x:x + w]  # Use `frame` instead of `gray` to keep 3 channels (RGB)
        face_roi = cv2.resize(face_roi, (64, 64))  # Resize to match model input
        face_roi = face_roi.astype("float32") / 255.0  # Normalize pixel values
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        print("Processed Face ROI shape:", face_roi.shape)  # Debugging

        # Predict emotion
        prediction = CNNmodel.predict(face_roi)

        # Predict emotion
        prediction = CNNmodel.predict(face_roi)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Draw rectangle around face and put emotion text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
