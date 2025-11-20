from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializer import ImageUploadSerializer
from PIL import Image

# END POINT
@api_view(['POST'])
# Parser for incoming data
@parser_classes([MultiPartParser, FormParser])
def scan_image(request):
    # POST
    serializer = ImageUploadSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    image = serializer.validated_data["image"]

    return Response({
        "filename": image.name,
        "size": image.size,
        "type": image.content_type
    })