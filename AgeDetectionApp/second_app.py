from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.properties import ObjectProperty

import cv2

class MyLayout(Widget):
    image = ObjectProperty(None)

class CamApp(App):

    def build(self):
        Window.size = (1200, 700)
        # self.img1 = Image()
        self.layout = MyLayout()
        # sidePanel = BoxLayout(orientation='vertical')
        # sidePanel.add_widget(Button(text='Choose model'))
        # sidePanel.add_widget(Button(text='Use camera'))
        # sidePanel.add_widget(Button(text='Load video'))
        # sidePanel.add_widget(Button(text='Load image folder'))
        # layout.add_widget(sidePanel)
        # layout.add_widget(self.img1)
        # # opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        cv2.imshow("CV2 Image", frame)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.layout.image.texture = texture1


if __name__ == '__main__':
    CamApp().run()
    # cv2.destroyAllWindows()
