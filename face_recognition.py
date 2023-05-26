import cv2
import numpy as np
import threading
from tensorflow import keras

# Load model
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
agemodel = keras.models.load_model('Models/regression_efficientNet_mp_5_08.h5') 
cap = cv2.VideoCapture(0)

# Shared variables
frame = None
boxes = []
lock = threading.Lock()
pause_detection = False

def process_and_predict(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (200, 200)) 

    ar = im.astype('float32')
    ar /= 255.0
    ar = ar.reshape(1, 200, 200, 1) 
    age = agemodel.predict(ar)

    return(f'Age: {age}')

def face_detection():
    global frame, boxes, pause_detection
    while True:
        with lock:
            if pause_detection:
                continue
        if frame is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            local_boxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    box = box.astype("int")
                    (x,y,w,h) = box
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    extracted_face = frame[y:h, x:w]

                    age_str = process_and_predict(extracted_face)
                    local_boxes.append((box,age_str))
            with lock:
                boxes = local_boxes

threading.Thread(target=face_detection, daemon=True).start()

while True:
    ret, frame = cap.read()
    with lock:
        for (box,age_str) in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, age_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):  # space bar pressed
        with lock:
            pause_detection = not pause_detection
cap.release()
cv2.destroyAllWindows()
