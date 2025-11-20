from rest_framework import serializers
from .models import user

# User Serializer
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = user
        fields = '__all__'

# Image Serializer
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()