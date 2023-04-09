# MTCNN-VGGFace2-InceptionResnetV1
Implementation of https://github.com/timesler/facenet-pytorch 


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

This code segment generates two new directories titled "matched_faces" and "mismatched_faces" within the project's root directory. Subsequently, it examines all the folders located in the "lfw" directory and categorizes them into the relevant directory based on their image count. Folders possessing a single image are transferred to the "mismatched_faces" directory, while those containing multiple images are relocated to the "matched_faces" directory. The creation of new directories is facilitated by the mkdir command, whereas the for loop iterates through the folders present in the "lfw" directory. The if statement evaluates the quantity of images in each folder, and the mv command relocates the folder to the respective directory.

<img width="483" alt="image" src="https://user-images.githubusercontent.com/118690399/230802129-9055742f-2f2c-40e8-9a9b-8e79e01bc268.png">


My Results testing this model on the LFW Dataset:
![Figure_1](https://user-images.githubusercontent.com/118690399/228976526-02a12f93-d466-45a7-ba4f-72159bc8907b.png)

With my results as such:

![Screenshot 2023-04-08 at 23 15 53](https://user-images.githubusercontent.com/118690399/230744740-471f5eae-a125-4247-b262-16a7b4a445b7.png)

By looking closely at a model's performance, we can understand how well it works in different ways, not just by checking accuracy and F1 scores. Sometimes, models might seem to work well, but they are actually making wrong predictions. I learned this the hard way. So, it's important to find and fix any mistakes, and a deep analysis can stop models from being praised only for a few good metrics while ignoring others.

I changed the LFW images to see how things like blurriness or filters might change how well the model works. Here are the results:
![Screenshot 2023-03-31 at 15 44 45](https://user-images.githubusercontent.com/118690399/229152869-009550fa-d653-4407-95c2-15813b7765b2.png)

![Screenshot 2023-03-31 at 15 45 08](https://user-images.githubusercontent.com/118690399/229152953-a283d2e6-7d69-461b-ad85-f26b17abf6f7.png)

![Screenshot 2023-03-31 at 15 45 24](https://user-images.githubusercontent.com/118690399/229153036-49a7fed0-73a9-47c0-8910-35560517e256.png)

![Screenshot 2023-03-31 at 15 45 50](https://user-images.githubusercontent.com/118690399/229153159-ca112730-7237-4fec-bf08-d7b25cae2a0e.png)

I think more research papers and analyses on various algorithms should offer more information on their model's performance under a wide range of situations. This would make it easier to compare models and use them more effectively in real-world applications. However, it's not an easy task since many parameters and factors need to be considered.

In my other repository, I have performed a similar analysis on the face transformer model from the repository: https://github.com/zhongyy/Face-Transformer
Please check out my results which may intrigue you if you are interested in using the transformer architecute for face recognition:
https://github.com/DrThomasCleary/Face_Transformer_Analysis

It becomes quite evident that the face transformer, despite achieving decent levels of accuracy, struggles with generalization and is likely overfitted when faced with images that are blurry or low resolution. 

If you have any questions, please email me at: Bauanrashid@hotmail.com

