[app]
title = ClothesApp
package.name = clothesapp
package.domain = org.example
source.dir = .
source.include_exts = py,kv,png,jpg,ttf
version = 0.1
orientation = portrait

requirements = python3, kivy, numpy, opencv-python-headless, requests

android.permissions = CAMERA

android.api = 31
android.minapi = 21
android.ndk = 25b

fullscreen = 0
