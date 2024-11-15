import os
import cv2
import numpy as np
import sys

def get_edge_mean_color(image):
    """Calculate the mean color of the edges of the image."""
    top_edge = image[0, :, :]
    bottom_edge = image[-1, :, :]
    left_edge = image[:, 0, :]
    right_edge = image[:, -1, :]

    edge_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge], axis=0)
    mean_color = np.median(edge_pixels,axis=0).astype(int).tolist()
    
    return mean_color

def resize_and_pad(image, target_size, static_border_color = False, static_color = [256, 256, 256]):
    """
    Resize the image to the target size while maintaining aspect ratio, then pad to make it square.
    Defaults to extracting the median border color and padding with that. If static_border_color=True,
    it will instead use the static_color argument (default: white).
    """
    old_height, old_width = image.shape[:2]
    ratio = min(target_size / old_height, target_size / old_width)
    
    new_width, new_height = int(old_width * ratio), int(old_height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    delta_width = target_size - new_width
    delta_height = target_size - new_height
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)

    if static_border_color:
        color = static_color
    else:
        color = get_edge_mean_color(resized_image)

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
                    save_dir = os.path.join(destination_folder, relative_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    save_path = os.path.join(save_dir, file)
                    cv2.imwrite(save_path, standardized_image)
                    print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    """
    Executed if script is run rather than imported.
    Command line arguments are positional as <source_folder> <destination_folder> <target_size>

    Example: python image_processor.py data/img/raw data/img/256 256
    """
    if len(sys.argv) != 4:
        print("Usage: python image_processor.py <source_folder> <destination_folder> <target_size>")
        sys.exit(1)

    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    target_size = int(sys.argv[3])

    process_images(source_folder, destination_folder, target_size)