from PIL import Image, ImageOps, ImageEnhance
import os


path_to_original_images = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/working/mismatched_faces'
path_to_filters = {
    'grayscale': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_grayscale',
    'sepia': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_sepia',
    'color_tint': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_color_tint',
    'invert': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_invert',
    'contrast': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_contrast',
    'brightness': '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_brightness'
}

def apply_sepia(image):
    width, height = image.size
    pixels = image.load() 

    for py in range(height):
        for px in range(width):
            r, g, b = image.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255
            if tg > 255:
                tg = 255
            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)

    return image

def apply_color_tint(image, color):
    return ImageOps.colorize(image.convert('L'), '#000000', color)

for folder in os.listdir(path_to_original_images):
    folder_path = os.path.join(path_to_original_images, folder)
    if os.path.isdir(folder_path):
        for filter_name, filter_path in path_to_filters.items():
            os.makedirs(os.path.join(filter_path, folder), exist_ok=True)

        for filename in os.listdir(folder_path):
            if not filename.startswith('.'):
                img_path = os.path.join(folder_path, filename)
                if not os.path.isdir(img_path):
                    img = Image.open(img_path)

                    grayscale = ImageOps.grayscale(img)
                    sepia = apply_sepia(img.copy())
                    color_tint = apply_color_tint(img.copy(), '#FFA500') # Change the color for the color tint here

                    filters = {
                        'grayscale': grayscale,
                        'sepia': sepia,
                        'color_tint': color_tint
                    }

                    for filter_name, filtered_image in filters.items():
                        filtered_image_path = os.path.join(path_to_filters[filter_name], folder, f"{filename[:-4]}_{filter_name}.jpg")
                        filtered_image.save(filtered_image_path)

for folder in os.listdir(path_to_original_images):
    folder_path = os.path.join(path_to_original_images, folder)
    if os.path.isdir(folder_path):
        for filter_name, filter_path in path_to_filters.items():
            os.makedirs(os.path.join(filter_path, folder), exist_ok=True)

        for filename in os.listdir(folder_path):
            if not filename.startswith('.'):
                img_path = os.path.join(folder_path, filename)
                if not os.path.isdir(img_path):
                    img = Image.open(img_path)

                    grayscale = ImageOps.grayscale(img)
                    sepia = apply_sepia(img.copy())
                    color_tint = apply_color_tint(img.copy(), '#FFA500') # Change the color for the color tint here
                    invert = ImageOps.invert(img)
                    contrast = ImageEnhance.Contrast(img).enhance(2.0) # Change the contrast factor here (2.0 means doubled contrast)
                    brightness = ImageEnhance.Brightness(img).enhance(1.5) # Change the brightness factor here (1.5 means 50% brighter)

                    filters = {
                        'grayscale': grayscale,
                        'sepia': sepia,
                        'color_tint': color_tint,
                        'invert': invert,
                        'contrast': contrast,
                        'brightness': brightness
                    }

                    for filter_name, filtered_image in filters.items():
                        filtered_image_path = os.path.join(path_to_filters[filter_name], folder, f"{filename[:-4]}_{filter_name}.jpg")
                        filtered_image.save(filtered_image_path)