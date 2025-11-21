from rest_framework import serializers
from .models import user

# User Serializer (Pake model)
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = user
        fields = '__all__'

# Image Serializer (Tidak pake model)
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()