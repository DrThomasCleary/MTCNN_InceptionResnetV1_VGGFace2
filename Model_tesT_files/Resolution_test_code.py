from PIL import Image
import os

path_to_original_images = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces'
path_to_lowres_images = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_48_res'
resolution = (48,48) # set the desired resolution here

for folder in os.listdir(path_to_original_images):
    folder_path = os.path.join(path_to_original_images, folder)
    if os.path.isdir(folder_path):
        os.makedirs(os.path.join(path_to_lowres_images, folder), exist_ok=True)
        for filename in os.listdir(folder_path):
            if not filename.startswith('.'):
                img_path = os.path.join(folder_path, filename)
                if not os.path.isdir(img_path):
                    img = Image.open(img_path)
                    lowres = img.resize(resolution)
                    lowres_path = os.path.join(path_to_lowres_images, folder, f"{filename[:-4]}_lowres.jpg")
                    lowres.save(lowres_path)