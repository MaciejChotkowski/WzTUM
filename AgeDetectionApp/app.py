from enum import Enum
# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import imghdr
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

import cv2
import os

import tensorflow as tf
import os
print("Num of GPUs available: ", len(tf.test.gpu_device_name()))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class InputType(Enum):
    VIDEO = 1
    IMAGE = 2
    CAMERA = 3


input_type = InputType.CAMERA
capture = cv2.VideoCapture(0)
agemodel = None


def process_and_predict(im):
    global agemodel
    # im = Image.open(file).convert('L')
    width, height = im.size
    if width == height:
        im = im.resize((200, 200), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.ANTIALIAS)

    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(1, 200, 200, 1)
    age = agemodel.predict(ar)

    age = np.argmax(age)

    if age == 0:
        print('Age: [1, 2]')
    elif age == 1:
        print('Age: [3, 9]')
    elif age == 2:
        print('Age: [10, 20]')
    elif age == 3:
        print('Age: [21, 27]')
    elif age == 4:
        print('Age: [28, 45]')
    elif age == 5:
        print('Age: [46, 65]')
    else:
        print('Age: (65, 80]')

    return im.resize((300, 300), Image.ANTIALIAS)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(Widget):
    image = ObjectProperty(None)

    def switch_to_camera(self):
        global input_type, capture
        capture.release()
        capture = cv2.VideoCapture(0)
        input_type = InputType.CAMERA

    def switch_to_video(self):
        global input_type
        capture.release()
        input_type = InputType.VIDEO
        content = LoadDialog(load=self.load_video, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose video file", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def switch_to_image(self):
        global input_type, capture
        capture.release()
        input_type = InputType.IMAGE
        content = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose images", content=content,
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
        if len(filename) > 0:
            load_file_path = os.path.join(filename[0])
        self.dismiss_popup()

    def load_video(self, filename):
        global capture
        if len(filename) > 0:
            load_file_path = os.path.join(filename[0])
            capture = cv2.VideoCapture(load_file_path)
        self.dismiss_popup()

    def load_image(self, filename):
        global capture
        if len(filename) > 0:
            load_file_path = os.path.join(filename[0])
            capture = cv2.VideoCapture(load_file_path)
        self.dismiss_popup()


class CamApp(App):

    def build(self):
        Window.size = (1200, 700)
        self.layout = Root()
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/30.0)

        # TO DO ZASTĄPIENIA OFC
        self.face_cascade = cv2.CascadeClassifier('../../haar.xml')

        return self.layout

    def update(self, dt):
        if input_type == InputType.CAMERA:
            self.display_camera()
        elif input_type == InputType.IMAGE:
            self.display_image()
        elif input_type == InputType.VIDEO:
            self.display_video()

    def display_camera(self):
        global capture
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.layout.image.texture = texture1
        # process_and_predict(texture1)

    def display_video(self):
        global capture
        ret, frame = capture.read()
        if ret == True:
            cv2.imshow('frame', frame)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.layout.image.texture = texture1

    def display_image(self):
        global capture
        ret, frame = capture.read()
        if ret == True:
            cv2.imshow('frame', frame)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.layout.image.texture = texture1


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()
