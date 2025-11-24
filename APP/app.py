import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
import numpy as np
import cv2
import requests

from utils.reccomend import recommend
from utils.parseImage import scan_folder

# Optional: Set a fixed window size for desktop testing
Window.size = (360, 640)

# Define each screen as a class
class HomeScreen(Screen):
    pass

class CameraScreen(Screen):
    def on_enter(self):
        # when opening this screen
        self.ids.camera_view.play = True  # Turn ON the camera

    def on_leave(self):
        # when leaving this screen
        self.ids.camera_view.play = False  # Turn OFF the camera

    def capture(self):
        camera = self.ids.camera_view
        # Get the texture (current frame)
        texture = camera.texture
        if not texture:
            print("No texture yet")
            return
        
        w,h = texture.size  # (width, height)
    
        # Get raw RGBA pixel data
        pixels = texture.pixels

        # Convert to NumPy array
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape(h, w, 4)  # Kivy uses RGBA format

        # Convert RGBA → BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Encode to JPEG in memory
        success, jpeg_bytes = cv2.imencode(".jpg", img_bgr)
        if not success:
            print("Failed to encode image!")
            return
        image_bytes = jpeg_bytes.tobytes()

        # Detect clothes and clothes color from image(API CALL post then send image_byte)
        url = "http://127.0.0.1:8000/api/scan/"
        files = {
            "image": ("camera.jpg", image_bytes, "image/jpeg")
        }
        r = requests.post(url, files=files)
        print("Response:\n" ,r.json())

        # Save di Android
        # # Save
        # if os.path.exists(f'img_list/{name}_{color_name}.jpg'):
        #     self.manager.current = "warn"
        # else:
        #     print("Screens available:", [s.name for s in self.manager.screens])
        #     # Kirim data ke screen selanjutnya
        #     self.manager.get_screen("confirm").recive_data(name, color_name, image)
        #     # Pindahkan screen
        #     self.manager.current = "confirm"

# Display if clothes already exist in inventory
class WarnClothes(Screen):
    pass

# Confirm to save your new clothes
class ConfirmNewClothes(Screen):
    clothes_name = StringProperty("")
    clothes_color = StringProperty("")
    clothes_image = None

    def recive_data(self, name, color_name, image):
        def update(dt):
            self.clothes_name = name
            self.clothes_color = color_name
            self.clothes_image = image
            self.ids.clothesDetail.text = f"Clohtes type: {name}\nClothes color: {color_name}"

        Clock.schedule_once(update, 0.1)  # delay 1 detik

    def saveImage(self):
        cv2.imwrite(f"img_list/{self.clothes_name}_{self.clothes_color}.jpg", self.clothes_image)
        self.manager.current = 'result'

class ResultScreen(Screen):
    def load_images(self):
        folder_path = "img_list"
        grid = self.ids.image_grid
        grid.clear_widgets()

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)

                item = BoxLayout(orientation='vertical')
                item.spacing = 10
                img_widget = Image(source=img_path, size_hint_y=None, height=200)
                img_label = Label(text=filename.split('.')[0])
                item.add_widget(img_widget)
                item.add_widget(img_label)
                grid.add_widget(item)
    
    def on_pre_enter(self):
        # Reload gallery every time you enter
        self.load_images()

class RecommendScreen(Screen):
    def on_pre_enter(self):
        data = scan_folder("img_list")
        self.ids.spinner.values = [f"{item} {color}" for item, color in data]
        

    def on_selected(self, value):
        self.ids.spinner.text = value

    def go_next(self):
        result = self.manager.get_screen("result_recommend")
        spinner_value = self.ids.spinner.text
        clothes_type, clothes_color = spinner_value.split(" ")
        item_list = self.ids.spinner.values
        result.recive_data(clothes_type, clothes_color, item_list)
        self.manager.current = "result_recommend"

class RecommendResultScreen(Screen):
    def recive_data(self, clothes_type, clothes_color, item_list):
        recommend_result = recommend(clothes_type, clothes_color, item_list)
        self.ids.recommend_label.text = recommend_result[0];


# Define Screen Manager
class ScreenManagement(ScreenManager):
    pass

class Application(App):
    def build(self):
        return Builder.load_file("app.kv") # load kv file

if __name__ == '__main__':
    Application().run()
