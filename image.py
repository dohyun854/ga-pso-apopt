import cv2
import numpy as np

def extract_all_wall_coordinates(image_path):
    # 1. 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 이진화 (Thresholding)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 3. 벽 픽셀 좌표 추출
    wall_coordinates = []
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 255:  # 흰색 픽셀일 경우
                wall_coordinates.append((x, y))
    
    return wall_coordinates