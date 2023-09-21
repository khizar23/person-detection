import torch
import cv2
import os
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Open the video
cap = cv2.VideoCapture('ytml.mp4')
# Create an output directory to save bounding box images

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Define the coordinates of the area (adjust these according to your video frame size)
area = [(1098,435), (1527, 435), (1098, 435), (1527, 435)]

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Perform object detection using the YOLOv5 model
    result = model(img)
    df = result.pandas().xyxy[0]

    for ind in df.index:
        label = df['name'][ind]
        # Check if the detected object is a person (you can customize this label)
        if label == 'person':
            x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
            x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])

            # Calculate the midpoint of the bounding box
            midpoint_x = (x1 + x2) // 2
            midpoint_y = (y1 + y2) // 2

            # Check if the midpoint of the bounding box passes through the area
            if any(cv2.pointPolygonTest(np.array(area), (midpoint_x, midpoint_y), False) >= 0 for (midpoint_x, midpoint_y) in
                   [(midpoint_x, midpoint_y)]):

                # Save the bounding box image to the output directory
                person_img = img[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f'person_{ind}.jpg')
                cv2.imwrite(output_path, person_img)

    cv2.imshow('frame',img)

# Release the video capture
cap.release()

print(f"Bounding box images saved in the '{output_dir}' folder.")