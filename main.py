# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# initializing MTCNN and InceptionResnetV1 
mtcnn = MTCNN(image_size=100, margin=24, keep_all=False, min_face_size=100)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# load the training dataset (insert training2ormore for better performance)
train_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/training2ormore')
train_loader = DataLoader(train_dataset, collate_fn=lambda x: x[0])

# load the test dataset (insert testing2ormore for better performance)
test_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/testing2ormore')
test_loader = DataLoader(test_dataset, collate_fn=lambda x: x[0])

# generate embeddings for the training dataset
train_name_list = []
train_embedding_list = []
for image, index in train_loader:
    face, face_prob = mtcnn(image, return_prob=True)
    if face is not None and face_prob > 0.99:
        emb = resnet(face.unsqueeze(0))
        train_embedding_list.append(emb.detach())
        train_name_list.append(train_dataset.classes[index])
    else:
        print("No face detected in image training file:", index)

# generate embeddings for the test dataset
test_name_list = []
test_embedding_list = []
for image, index in test_loader:
    face, face_prob = mtcnn(image, return_prob=True)
    if face is not None and face_prob > 0.99:
        emb = resnet(face.unsqueeze(0))
        test_embedding_list.append(emb.detach())
        test_name_list.append(test_dataset.classes[index])
        
    else:
        print("No face detected in image testing file:", index)

# calculate the minimum distance between the embeddings of the detected faces and the embeddings of the known faces
def minimum_distance(embedding_list, emb):
    dist_list = []
    for emb_db in embedding_list:
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    if len(dist_list) > 0:
        min_dist = min(dist_list)
        min_dist_index = dist_list.index(min_dist)
        name = train_name_list[min_dist_index]
        return name, min_dist, dist_list
    else:
        return 'Unknown', None, []

# calculate the accuracy of the model
y_true = []
y_pred = []
dist_list = []
for i, emb in enumerate(test_embedding_list):
    name_true = test_name_list[i]
    name_pred, min_dist, distances = minimum_distance(train_embedding_list, emb)
    y_true.append(name_true)
    y_pred.append(name_pred)
    dist_list.append(min_dist)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
avg_min_dist = sum(dist_list) / len(dist_list)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
print("Average minimum distance: ", avg_min_dist)