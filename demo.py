import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('demo_model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emoji_dict = {
  0: "emoji/angry-face.png",
  1: "emoji/nauseated-face.png",
  2: "emoji/face-screaming-in-fear.png",
  3: "emoji/smiling-face-with-smiling-eyes.png",
  4: "emoji/sad-but-relieved-face.png",
  5: "emoji/face-screaming-in-fear.png",
  6: "emoji/neutral-face.png",
}

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (64, 64)), -1), 0)
        cropped_img = cropped_img.astype('float64')
        cropped_img = cropped_img/255.
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.imread(emoji_dict[maxindex])
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        img_height, img_width, _ = img.shape
        roi = frame[-img_height-10:-10, -img_width-10:-10]
        roi[np.where(mask)] = 0
        roi += img

    cv2.imshow('Video', cv2.resize(frame,(1280,720),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
