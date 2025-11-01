from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window

# Optional: Set a fixed window size for desktop testing
Window.size = (360, 640)

# Define each screen as a class
class HomeScreen(Screen):
    pass

class CameraScreen(Screen):
    def capture(self):
        print("Capturing image... (implement your camera code here)")

class ResultScreen(Screen):
    def on_enter(self):
        print("Showing results...")

# ScreenManager to handle transitions
class ScreenManagement(ScreenManager):
    pass

# Load the KV file
kv = Builder.load_file("main.kv")

class MultiPageApp(App):
    def build(self):
        return kv

if __name__ == '__main__':
    MultiPageApp().run()
