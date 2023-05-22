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

from enum import Enum


class InputType(Enum):
    VIDEO = 1
    IMAGE = 2
    CAMERA = 3


input_type = InputType.CAMERA


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(Widget):
    image = ObjectProperty(None)

    def switch_to_camera(self):
        global input_type
        input_type = InputType.CAMERA

    def switch_to_video(self):
        global input_type
        input_type = InputType.VIDEO
        content = LoadDialog(load=self.load_video, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose video file", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def switch_to_image(self):
        global input_type
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

        # WCZYTYWANIE MODELU

    def load_video(self, filename):
        if len(filename) > 0:
            load_file_path = os.path.join(filename[0])
        self.dismiss_popup()

        # WCZYTYWANIE FILMU

    def load_image(self, filename):
        if len(filename) > 0:
            load_file_path = os.path.join(filename[0])
        self.dismiss_popup()

        # WCZYTYWANIE OBRAZKÓW


class CamApp(App):

    def build(self):
        Window.size = (1200, 700)
        self.layout = Root()
        self.capture = cv2.VideoCapture(0)
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
        ret, frame = self.capture.read()
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

    def display_video(self):
        pass

    def display_image(self):
        pass


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()
