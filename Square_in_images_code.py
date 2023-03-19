import os
import random
from PIL import Image, ImageDraw

input_folder = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/matched_faces'
output_folder = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/matched_faces_black_square_15%'

def add_black_square(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Set the square size to 10% of the image's smaller dimension
    square_size = int(min(width, height) * 0.15)

    x_pos = (width - square_size) // 2
    y_pos = (height - square_size) // 2

    draw.rectangle([x_pos, y_pos, x_pos + square_size, y_pos + square_size], fill='black')
    return image

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add other file formats if needed
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            occluded_image = add_black_square(image)

            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            occluded_image.save(os.path.join(output_subfolder, filename))