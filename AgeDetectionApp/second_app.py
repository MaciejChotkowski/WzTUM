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

    def switch_to_video(self):
        global input_type
        input_type = InputType.VIDEO

    def switch_to_camera(self):
        global input_type
        input_type = InputType.CAMERA

    def switch_to_image(self):
        global input_type
        input_type = InputType.IMAGE

    def load_model(self):
        self.show_load()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.text_input.text = stream.read()

        self.dismiss_popup()


class CamApp(App):

    def build(self):
        Window.size = (1200, 700)
        self.layout = Root()
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.layout

    def update(self, dt):
        if input_type == InputType.CAMERA:
            ret, frame = self.capture.read()
            cv2.imshow("CV2 Image", frame)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.layout.image.texture = texture1
        elif input_type == InputType.IMAGE:
            pass
        elif input_type == InputType.VIDEO:
            pass


if __name__ == '__main__':
    CamApp().run()
    # cv2.destroyAllWindows()
