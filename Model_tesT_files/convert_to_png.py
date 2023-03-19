import os
from PIL import Image

input_root = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces'
output_root = '/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces_png'

if not os.path.exists(output_root):
    os.makedirs(output_root)

# Iterate through all subfolders in the input root folder
for subdir, _, _ in os.walk(input_root):
    output_subdir = os.path.join(output_root, os.path.relpath(subdir, input_root))

    # Create corresponding subfolders in the output root folder
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Iterate through all the JPG files in the input subfolder
    for filename in os.listdir(subdir):
        if filename.lower().endswith('.jpg'):
            # Open the JPG image
            img = Image.open(os.path.join(subdir, filename))

            # Convert the file extension from .jpg to .png
            png_filename = os.path.splitext(filename)[0] + '.png'

            # Save the image in PNG format in the corresponding output subfolder
            img.save(os.path.join(output_subdir, png_filename))

print("All JPG images have been converted to PNG.")