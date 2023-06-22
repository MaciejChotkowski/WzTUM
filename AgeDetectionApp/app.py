import os
import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from datetime import datetime

# Load model
modelFile = "../Models/face-detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "../Models/face-detection/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

capture = cv2.VideoCapture(0)
agemodel = None

output_path = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)


class Writer:
    def __init__(self):
        self.type = None
        self.path = None
        self.output = None
        self.writing = False

    def set(self, type, path, fps=0, width=0, height=0):
        self.writing = True
        self.type = type
        self.path = path
        if self.type == 'video':
            self.output = cv2.VideoWriter(path,
                                          cv2.VideoWriter_fourcc(*'MJPG'),
                                          fps, (width, height))

    def write(self, frame):
        if self.type == 'video':
            self.output.write(frame)
        elif self.type == 'image':
            cv2.imwrite(self.path, frame)

    def release(self):
        self.writing = False
        if self.type == 'video':
            self.output.release()


writer = Writer()


def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    box = box.astype("int")
                    (x, y, x2, y2) = box.astype("int")
                    r = (x2-x)-(y2-y)
                    if(r>0):
                        m = (y+y2)//2
                        y = m - (x2-x)//2
                        y2 = m + (x2-x)//2
                    elif(r<0):
                        m = (x+x2)//2
                        x= m - (y2-y)//2
                        x2 = m + (y2-y)//2
                    
                    x -= (x2-x)//20
                    x2 +=(x2-x)//20
                    y -= (y2-y)//20
                    y2 +=(y2-y)//20
                    if(x<0 or y<0):
                        continue
                    extracted_face = frame[y:y2, x:x2]
                    
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    #cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    age_str = process_and_predict(extracted_face)

                    text_y = y-10 if y + (y2-y)/2 > int(frame.shape[0]/2) else y2+25
                    cv2.putText(frame, age_str, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame


def process_and_predict(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (200, 200)) 

    ar = im.astype('float32')
    ar /= 255.0
    ar = ar.reshape(1, 200, 200, 1) 
    age = agemodel.predict(ar)

    return f'Age: {int(age)}'


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(Widget):
    image = ObjectProperty(None)
    age_label = ObjectProperty(None)

    def switch_to_camera(self):
        global input_type, capture
        capture.release()
        capture = cv2.VideoCapture(0)

    def switch_to_video(self):
        global input_type
        capture.release()
        content = LoadDialog(load=self.load_video, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose video file", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def switch_to_image(self):
        global input_type, capture
        capture.release()
        content = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose image", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def switch_model(self):
        content = LoadDialog(load=self.load_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose model file", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def load_model(self, filename):
        global agemodel
        if len(filename) > 0:
            load_file_path = filename[0]
            agemodel = keras.models.load_model(load_file_path)
        self.dismiss_popup()

    def load_video(self, filename):
        global capture
        global writer
        if len(filename) > 0:
            load_file_path = filename[0]
            filename = os.path.basename(load_file_path)
            new_filename = os.path.join(
                'output', filename[:filename.rfind('.')] + '_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S") + '.avi')
            capture = cv2.VideoCapture(load_file_path)
            writer.set(type='video', path=new_filename, fps=int(
                capture.get(5)), width=int(capture.get(3)), height=int(capture.get(4)))
        self.dismiss_popup()

    def load_image(self, filename):
        global capture
        global writer
        if len(filename) > 0:
            load_file_path = filename[0]
            capture = cv2.VideoCapture(load_file_path)
            filename = os.path.basename(load_file_path)
            new_filename = os.path.join(
                'output', filename[:filename.rfind('.')] + '_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S") + '.jpg')
            print(new_filename)
            writer.set(type='image', path=new_filename)
        self.dismiss_popup()


class CamApp(App):

    def build(self):
        Window.size = (1200, 700)
        self.title = 'Age detection app'
        self.layout = Root()
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/30.0)

        return self.layout

    def close_app(self):
        self.stop()

    def update(self, dt):
        self.display()

    def display(self):
        global capture
        global writer
        ret, frame = capture.read()

        if ret:
            if agemodel is not None:
                frame = detect_faces(frame)
                if writer.writing:
                    writer.write(frame)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.layout.image.texture = texture1

        else:
            if writer.writing:
                writer.release()


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()
