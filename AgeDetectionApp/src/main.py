from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os

root = Tk()
root.title("AgeDetectionApp")


def image():
    image_list = []
    valid_images = [".jpg",".jpeg",".png"]
    selected_folder = filedialog.askdirectory(initialdir="/", title="Select a directory")
    for f in os.listdir(selected_folder):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
    image_list.append(Image.open(os.path.join(selected_folder, f)))

    image_label = Label(text=image_list[0])
    image_label.grid(row = 0, column=0, columnspan=4)
    return

def camera():
    return

def video():
    return


## Buttons
button_model = Button(root, text="Choose model")
button_camera = Button(root, text="Use camera")
button_image = Button(root, text="Choose images folder", command=image)
button_video = Button(root, text="Choose video file")

button_model.grid(row=1, column=0)
button_camera.grid(row=1, column=1)
button_image.grid(row=1, column=2)
button_video.grid(row=1, column=3)


root.mainloop()