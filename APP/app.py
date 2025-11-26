import os
import io
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
import requests

from utils.reccomend import recommend
from utils.parseImage import scan_folder

# Fix window size on desktop (ignored on Android)
Window.size = (360, 640)


# === SCREENS ===

class HomeScreen(Screen):
    pass


class CameraScreen(Screen):
    def on_enter(self):
        self.ids.camera_view.play = True

    def on_leave(self):
        self.ids.camera_view.play = False

    def capture(self):
        camera = self.ids.camera_view
        texture = camera.texture

        if texture is None:
            print("Camera not ready yet")
            return

        buffer = io.BytesIO()
        texture.save(buffer, flipped=False, fmt='png')  # Kivy provides PNG saving
        image_bytes = buffer.getvalue()

        # Send to your API
        url = "https://object-detectionoutfitmodel-production.up.railway.app/api/scan/"
        try:
            response = requests.post(url, files={"image": ("camera.jpg", image_bytes, "image/jpeg")})
            print(response.json())
        except:
            print("API request failed (no internet?)")

        # Pass to next screen
        # self.manager.get_screen("confirm").recive_data(...)
        # self.manager.current = "confirm"


class WarnClothes(Screen):
    pass


class ConfirmNewClothes(Screen):
    clothes_name = StringProperty("")
    clothes_color = StringProperty("")
    clothes_image = None

    def recive_data(self, name, color_name, image):
        def update(dt):
            self.clothes_name = name
            self.clothes_color = color_name
            self.clothes_image = image
            self.ids.clothesDetail.text = f"Type: {name}\nColor: {color_name}"

        Clock.schedule_once(update, 0.1)

    def saveImage(self):
        save_path = f"img_list/{self.clothes_name}_{self.clothes_color}.jpg"
        self.clothes_image.save(save_path)
        self.manager.current = "result"


class ResultScreen(Screen):
    def load_images(self):
        folder = "img_list"
        grid = self.ids.image_grid
        grid.clear_widgets()

        os.makedirs(folder, exist_ok=True)

        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, f)

                box = BoxLayout(orientation="vertical")
                img = Image(source=path, size_hint_y=None, height=200)
                lbl = Label(text=f.split(".")[0])

                box.add_widget(img)
                box.add_widget(lbl)
                grid.add_widget(box)

    def on_pre_enter(self):
        self.load_images()


class RecommendScreen(Screen):
    def on_pre_enter(self):
        data = scan_folder("img_list")
        self.ids.spinner.values = [f"{a} {b}" for a, b in data]

    def on_selected(self, v):
        self.ids.spinner.text = v
        self.ids.confirm.opacity = 1
        self.ids.confirm.disabled = False

    def go_next(self):
        val = self.ids.spinner.text
        clothes_type, clothes_color = val.split(" ")

        result = self.manager.get_screen("result_recommend")
        result.recive_data(clothes_type, clothes_color, self.ids.spinner.values)
        self.manager.current = "result_recommend"


class RecommendResultScreen(Screen):
    def recive_data(self, clothes_type, clothes_color, item_list):
        res = recommend(clothes_type, clothes_color, item_list)
        self.ids.recommend_label.text = res[0]


class ScreenManagement(ScreenManager):
    pass


class Application(App):
    def build(self):
        return Builder.load_file("app.kv")


if __name__ == "__main__":
    Application().run()
