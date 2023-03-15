#importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import numpy as np

# initializing MTCNN and InceptionResnetV1 
mtcnn = MTCNN(image_size=100, margin=24, keep_all=False, min_face_size=50)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# load the dataset
matched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/matched_faces')
matched_dataset.idx_to_class = {i: c for c, i in matched_dataset.class_to_idx.items()}

mismatched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces')
mismatched_dataset.idx_to_class = {i: c for c, i in mismatched_dataset.class_to_idx.items()}
mismatched_loader = DataLoader(mismatched_dataset, collate_fn=lambda x: x[0])

# generate embeddings for the dataset
matched_embedding_list = []
matched_name_list = []
for folder in os.listdir(matched_dataset.root):
    folder_path = os.path.join(matched_dataset.root, folder)
    if os.path.isdir(folder_path):
        # load the first two images in the folder
        images = []
        for i, filename in enumerate(sorted(os.listdir(folder_path))):
            if i >= 2:
                break
            image_path = os.path.join(folder_path, filename)
            image = datasets.folder.default_loader(image_path)
            images.append(image)
        # detect faces and generate embeddings for the two images in the folder
        if len(images) == 2:
            embeddings = []
            for i in range(2):
                face, face_prob = mtcnn(images[i], return_prob=True)
                if face is not None and face_prob > 0.73:
                    emb = resnet(face.unsqueeze(0))
                    embeddings.append(emb.detach())
                else:
                    print(f"No face detected in {os.path.basename(image_path)}")
                    break
            if len(embeddings) == 2:
                matched_embedding_list.extend(embeddings)
                matched_name_list.extend([folder] * 2)
            else:
                print(f"Not enough faces detected in {folder_path}")

# generate embeddings for the dataset
mismatched_embedding_list = []
mismatched_name_list = []
for image, index in mismatched_loader:
    face, face_prob = mtcnn(image, return_prob=True)
    if face is not None and face_prob > 0.73:
        emb = resnet(face.unsqueeze(0))
        mismatched_embedding_list.append(emb.detach())
        mismatched_name_list.append(mismatched_dataset.idx_to_class[index])
    else:
        print("No face detected in image:", index)

if len(mismatched_embedding_list) % 2 != 0:
    mismatched_embedding_list.pop()
    mismatched_name_list.pop()

# calculate the accuracy and F1 score of the model
num_correct = 0
dist_list = []
y_true = []
y_pred = []

# compare embeddings and calculate metrics
for i in range(0, len(matched_embedding_list), 2):
    emb1 = matched_embedding_list[i] 
    emb2 = matched_embedding_list[i+1]
    dist = torch.dist(emb1, emb2).item()
    dist_list.append(dist)
    is_match = matched_name_list[i] == matched_name_list[i+1]
    y_true.append(is_match)
    y_pred.append(dist < 1.6429632902145386)
    if (dist < 1.6429632902145386 and is_match):
        num_correct += 1

for i in range(len(mismatched_embedding_list)):
    if i % 2 == 0:
        emb1 = mismatched_embedding_list[i] 
        emb2 = mismatched_embedding_list[i + 1]
        dist = torch.dist(emb1, emb2).item()
        dist_list.append(dist)
        is_match = mismatched_name_list[i] != mismatched_name_list[i + 1]
        y_true.append(is_match)
        y_pred.append(dist > 0.6010755896568298)
        if (dist < 0.6010755896568298 and not is_match):
            num_correct += 1

dist_matched = [torch.dist(matched_embedding_list[i], matched_embedding_list[i + 1]).item() for i in range(0, len(matched_embedding_list), 2)]
dist_mismatched = [torch.dist(mismatched_embedding_list[i], mismatched_embedding_list[i + 1]).item() for i in range(0, len(mismatched_embedding_list), 2)]

distances = dist_matched + dist_mismatched
labels = [1] * len(dist_matched) + [0] * len(dist_mismatched)

epsilon = 1e-8
precision, recall, thresholds = precision_recall_curve(labels, distances, pos_label=1)
f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)

fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=1)
roc_auc = auc(fpr, tpr)

epsilon = 1e-8
TP = sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
FP = sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
FN = sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
TN = sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP + epsilon)
recall = TP / (TP + FN + epsilon)
f1 = 2 * (precision * recall) / (precision + recall + epsilon)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("AUC: ", roc_auc)