# MTCNN-VGGFace2-InceptionResnetV1
Face recognition using MTCNN(face detection)-VGGFace2/InceptionResnetV1(face recognition)
To test any dataset, you need to have a folder directory as such:
Matched_Faces_folder
  |-Folder_of_Person_1
    |-Image_1_of_person_1
    |-Image_2_of_person_1

Mismatched_Faces_Folder
  |-Folder_of_Person_2
    |-Image_1_of_person_2
  |-Folder_of_Person_3
    |-Image_1_of_person_3
    
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
-Optimal Accuracy Threshold Distance: 0.9401680336067213
-Average Computation time: 0.07874587009719577
-Total images not recognised in matched_faces: 0
-Total images not recognised in mismatched_faces: 0
-Detection Rate: 100.0
-True Positives:  1298
-True Negatives:  1915
-False Negatives:  382
-False Positives:  119
-Average distance for matched faces: 0.8175285885199195
-Average distance for mismatched faces: 1.1436471557546857
-EER: 0.1494319778058716
-Accuracy:  0.8651050080775444
-Precision:  0.9160197600564574
-Recall:  0.7726190476190476
-F1 score:  0.8382305456893769

