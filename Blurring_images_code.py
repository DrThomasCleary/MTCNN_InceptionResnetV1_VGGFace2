from PIL import Image, ImageFilter
import os

path_to_original_images = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces'
path_to_blurred_images = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_blurred_1'

# blur amount
blur_amounts = [1]

for folder in os.listdir(path_to_original_images):
    folder_path = os.path.join(path_to_original_images, folder)
    if os.path.isdir(folder_path):
        os.makedirs(os.path.join(path_to_blurred_images, folder), exist_ok=True)
        for filename in os.listdir(folder_path):
            if not filename.startswith('.'):
                img_path = os.path.join(folder_path, filename)
                if not os.path.isdir(img_path):
                    img = Image.open(img_path)
                    for amount in blur_amounts:
                        blurred = img.filter(ImageFilter.GaussianBlur(amount))
                        blurred_path = os.path.join(path_to_blurred_images, folder, f"{filename[:-4]}_blur_{amount}.jpg")
                        blurred.save(blurred_path)