import cv2
import numpy as np

def extract_all_wall_coordinates(image_path):
   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    wall_coordinates = []
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 255: 
                wall_coordinates.append((x, y))
    
    return wall_coordinates

image_path = 'path_to_your_image.png'
wall_coordinates = extract_all_wall_coordinates(image_path)
print("벽의 모든 픽셀 좌표:", wall_coordinates)