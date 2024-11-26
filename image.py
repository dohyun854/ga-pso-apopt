import cv2
import numpy as np

def preprocess_image(image_path):
    """이미지를 이진화하고 내부와 외부를 분리합니다."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def extract_wall_and_internal_coordinates(image_path):
    """벽과 내부 좌표를 추출하여 반환합니다."""
    binary_image = preprocess_image(image_path)

    # 외부의 모든 좌표를 추출합니다.
    wall_coordinates = []
    internal_coordinates = []

    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 255:  # 벽
                wall_coordinates.append((x, y))
            else:  # 내부
                internal_coordinates.append((x, y))

    return wall_coordinates, internal_coordinates