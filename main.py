# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import random

# initializing MTCNN and InceptionResnetV1 
mtcnn = MTCNN(image_size=100, margin=24, keep_all=False, min_face_size=50)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# load the dataset
dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/lfw')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0], shuffle=True, batch_size=100)

# generate embeddings for the dataset
embedding_list = []
name_list = []
for image, index in loader:
    face, face_prob = mtcnn(image, return_prob=True)
    if face is not None and face_prob > 0.90:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(dataset.idx_to_class[index])
    else:
        print("No face detected in image:", index)

# calculate the accuracy and F1 score of the model
num_pairs = 50
num_correct = 0
dist_list = []
y_true = []
y_pred = []


for i in range(num_pairs*2):
    idx1 = random.randint(0, len(embedding_list)-1)
    idx2 = random.randint(0, len(embedding_list)-1)
    emb1 = embedding_list[idx1] 
    emb2 = embedding_list[idx2]
    dist = torch.dist(emb1, emb2).item()
    dist_list.append(dist)
    print(idx1)
    print(idx2)
    print(i)
    print(torch.equal(emb1, emb2))
    is_match = name_list[idx1] == name_list[idx2]
    y_true.append(is_match)
    y_pred.append(dist < 1.0)
    if (dist < 1.0 and is_match) or (dist > 1.0 and not is_match):
        num_correct += 1

print(y_true)
print(y_pred)        
accuracy = num_correct / num_pairs / 2
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# calculate ROC curve and area under the curve (AUC)
fpr, tpr, _ = roc_curve(y_true, dist_list)
auc_score = auc(fpr, tpr)

print(dist_list)
print(fpr)
print(tpr)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("AUC: ", auc_score)