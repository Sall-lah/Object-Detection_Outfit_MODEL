import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
import time

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
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png(f"img_list/IMG_{timestr}.png")
        print("Captured image!")

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
                img_widget = Image(source=img_path, size_hint_y=None, height=200)
                grid.add_widget(img_widget)
    
    def on_pre_enter(self):
        # Reload gallery every time you enter
        self.load_images()

# ScreenManager to handle transitions
class ScreenManagement(ScreenManager):
    pass

class Application(App):
    def build(self):
        root = Builder.load_file("main.kv") # load kv file
        return root

if __name__ == '__main__':
    Application().run()
