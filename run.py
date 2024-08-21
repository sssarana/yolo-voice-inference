import torch
import os
import time
from PIL import Image

# Model
model = torch.hub.load("C:\\Users\\Sofiia\\yolov5", "custom", "yolov5s.pt", source="local")

# Folder containing images
folder_path = "C:\\Users\\Sofiia\\Desktop\\images100"

# List all images in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

total_time = 0
total_confidence = 0
total_detections = 0
num_images = len(image_files)

i=0
# Inference and Results
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = Image.open(img_path)
    
    start_time = time.time()
    results = model(img)
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_time += inference_time
    
    confidences = results.xyxy[0][:, 4].numpy()  
    total_confidence += confidences.sum()
    total_detections += len(confidences)
    
    print(f"Results for {img_file}:")
    results.print()
    if i == 100:
        break
    #results.show()

if total_detections > 0:
    average_confidence = total_confidence / total_detections
else:
    average_confidence = 0

average_speed = total_time / num_images

print(f"Total time: {total_time:.4f} seconds")
print(f"Average Inference Speed: {average_speed:.4f} seconds per image")
print(f"Average Confidence: {average_confidence:.4f}")
