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

This block of code creates two new directories named "matched_faces" and "mismatched_faces" in the root directory of the project. It then searches for all folders in the "lfw" directory and sorts them into the appropriate directory based on the number of images they contain. Folders with only one image are moved to the "mismatched_faces" directory, while folders with more than one image are moved to the "matched_faces" directory. The mkdir command is used to create the new directories, while the for loop iterates through the folders in the "lfw" directory. The if statement checks how many images each folder contains, and the mv command moves the folder to the corresponding directory.


My Results testing this model on the LFW Dataset:
![Figure_1](https://user-images.githubusercontent.com/118690399/228976526-02a12f93-d466-45a7-ba4f-72159bc8907b.png)

With my results as such:

![Screenshot 2023-03-30 at 23 31 48](https://user-images.githubusercontent.com/118690399/228978090-9770ad6b-f9fd-4b3c-a0bd-2844113bdd65.png)


With a detailed analysis such as this, you can accurately assess how well a model is performing in all aspects, rather than relying solely on accuracy and F1 scores. Often, models can be deceptive in appearing to perform well, but in reality, they are completely incorrect. I learned this lesson the hard way. Therefore, it is crucial to eliminate any potential areas of error, and a thorough analysis can help prevent models from receiving recognition based solely on a select few metrics, while neglecting others.
