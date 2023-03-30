# MTCNN-VGGFace2-InceptionResnetV1
Face recognition using MTCNN(face detection)-VGGFace2/InceptionResnetV1(face recognition)
To test any dataset, you need to have a folder directory as such:

![Screenshot 2023-03-30 at 23 31 14](https://user-images.githubusercontent.com/118690399/228978008-9fd910cf-d18f-402e-854b-ae16cd52f40e.png)


This way The program will compare the images of person 1 to see if it is the right face or not, and then take image 1 of person 2 and 3 to see if 
it rightly predicts that these 2 people are not the same person. 

You can thus test this model on any dataset. For example, I used this code below to split the "LFW_deepfunneled" Dataset which you can download online 
and split it into matched and mismatched folders as such:

mkdir /path_to_directory/{matched_faces,mismatched_faces} && \
for folder in path_to_directory/lfw/*; do \
  num_files=$(find "$folder" -maxdepth 1 -type f | wc -l); \
  if [[ $num_files -eq 1 ]]; then \
    mv "$folder" path_to_directory/mismatched_faces/; \
  else \
    mv "$folder" path_to_directory/matched_faces/; \
  fi; \
done

This code snippet creates two new directories in the root of the project folder named "matched_faces" and "mismatched_faces".
It then iterates over each folder in the "lfw" directory and moves each folder containing only one image to the "mismatched_faces" directory, 
and each folder containing more than one image to the "matched_faces" directory. The mkdir command creates the new directories, 
and the for loop iterates over the folders in the "lfw" directory. 
The if statement checks whether a folder contains only one image or more than one image, 
and the mv command moves the folder to the appropriate directory.


My Results testing this model on the LFW Dataset:
![Figure_1](https://user-images.githubusercontent.com/118690399/228976526-02a12f93-d466-45a7-ba4f-72159bc8907b.png)

With my results as such:

![Screenshot 2023-03-30 at 23 31 48](https://user-images.githubusercontent.com/118690399/228978090-9770ad6b-f9fd-4b3c-a0bd-2844113bdd65.png)



