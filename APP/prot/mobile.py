import numpy as np
import onnxruntime
import cv2
from PIL import Image

# Load the ONNX model
session = onnxruntime.InferenceSession("prot/best.onnx")

# Define the input shape expected by the model
input_shape = (1, 3, 640, 640)

def preprocess_image(image_path):
    # Load the image
    image = np.array(Image.open(image_path))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the input shape expected by the model
    resized_image = cv2.resize(gray_image, input_shape[2:0:-1])

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Add a batch dimension
    input_data = np.expand_dims(normalized_image, axis=0)

    return input_data

def postprocess_output(output):
    # Get the bounding box predictions
    boxes = output[0][:, :4]
    scores = output[0][:, 4]

    # Filter out boxes with low confidence scores
    filtered_boxes = boxes[scores > 0.5]

    # Convert the bounding boxes from relative to absolute coordinates
    height, width = input_shape[2:0:-1]
    absolute_boxes = filtered_boxes * np.array([height, width, height, width])

    # Plot the bounding boxes on the original image
    image = cv2.imread(input_data)
    for box in absolute_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the input image
input_image = Image.open("img_list/tshirt_gray.jpg")

# Preprocess the image
input_data = preprocess_image(input_image)

# Run the model on the input data
output = session.run(None, {"input": input_data})

# Postprocess the output and plot the bounding boxes
postprocess_output(output)