import cv2
from PIL import Image
import numpy as np
from tensorflow import keras

# Load model
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

agemodel = keras.models.load_model('Models\simpleagepredictionmodel.h5') # nasz model

def process_and_predict(file):
    im = Image.open(file).convert('L')
    width, height = im.size
    if width == height:
        im = im.resize((200,200), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((200,200), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((200,200), Image.ANTIALIAS)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(1, 200, 200, 1) # caly resizing moze tez mozna prosciej w cv2, bez zapisywania pliku
    age = agemodel.predict(ar)
    
    age = np.argmax(age)
    
    if age == 0: # to trzeba zmienic, jak beda inne przedzialy, albo liczba po prostu
        return('Age: [1, 2]')
    elif age == 1:
        return('Age: [3, 9]')
    elif age == 2:
        return('Age: [10, 20]')
    elif age == 3:
        return('Age: [21, 27]')
    elif age == 4:
        return('Age: [28, 45]')
    elif age == 5:
        return('Age: [46, 65]')
    else:
        return('Age: (65, 80]')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")

            x = max(0, x - 20)  # subtract from the x-coordinate (left bound)
            y = max(0, y - 20)  # subtract from the y-coordinate (top bound)
            w = min(frame.shape[1], w + 20)  # add to the w-coordinate (right bound)
            h = min(frame.shape[0], h + 20)  # add to the h-coordinate (bottom bound)

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            extracted_face = frame[y:h, x:w]
            cv2.imwrite('extracted_face.png', extracted_face)

            age_str = process_and_predict("extracted_face.png")
            cv2.putText(frame, age_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()