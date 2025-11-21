from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import cv2
import numpy as np
import utils.runModel as runModel

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def scan_image(request):
    # 1. Get image from request
    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No image provided"}, status=400)

    # 2. Read file into memory (NO SAVING)
    image_bytes = image_file.read()

    # 3. Convert to numpy array (OpenCV friendly)
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # 4. Decode into OpenCV image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR format

    # --- Example: Do processing ---
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    x1, y1, x2, y2, name, color_name = runModel.detect(img_bgr);

    # (you can run ML model here)
    # example: result = model.predict(image)

    return Response({
        "message": "Image processed successfully.",
        "result": {
            "name": name,
            "color": color_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    })