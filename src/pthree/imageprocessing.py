import os
import cv2
import numpy as np

def resize_and_pad(image, target_size, color = [256, 256, 256]):
    """
    Resize the image to the target size while maintaining aspect ratio, then pad to make it square.
    Defaults to white background, should be modified to extract average border color.
    """
    old_height, old_width = image.shape[:2]
    ratio = min(target_size / old_height, target_size / old_width)
    
    new_width, new_height = int(old_width * ratio), int(old_height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    delta_width = target_size - new_width
    delta_height = target_size - new_height
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return padded_image

def process_images(source_folder, destination_folder, target_size):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    standardized_image = resize_and_pad(image, target_size)
                    
                    relative_path = os.path.relpath(root, source_folder)
                    save_dir = os.path.join(destination_folder + "_" + str(target_size), relative_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    save_path = os.path.join(save_dir, file)
                    cv2.imwrite(save_path, standardized_image)
                    print(f"Processed and saved: {save_path}")