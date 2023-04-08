#importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random 

# initializing MTCNN and InceptionResnetV1 
mtcnn = MTCNN(image_size = 112)
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# load the dataset
matched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/Square_test/matched_faces_black_square_20%')
matched_loader = DataLoader(matched_dataset, collate_fn=lambda x: x[0])

mismatched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/Square_test/mismatched_faces_black_square_20%')
mismatched_dataset.idx_to_class = {i: c for c, i in mismatched_dataset.class_to_idx.items()}
mismatched_loader = DataLoader(mismatched_dataset, collate_fn=lambda x: x[0])

def calculate_accuracy(labels, distances, threshold):
    correct_predictions = 0
    total_predictions = len(labels)

    for i in range(len(distances)):
        if labels[i] == 1 and distances[i] <= threshold:
            correct_predictions += 1
        if labels[i] == 0 and distances[i] > threshold:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy
def calculate_far_frr(labels, distances, threshold):
    num_positive = np.sum(labels)
    num_negative = len(labels) - num_positive
    false_accepts = 0
    false_rejects = 0

    for i in range(len(distances)):
        if labels[i] == 1 and distances[i] > threshold:
            false_rejects += 1
        if labels[i] == 0 and distances[i] <= threshold:
            false_accepts += 1

    FAR = false_accepts / num_negative
    FRR = false_rejects / num_positive
    return FAR, FRR

# Initialize variables to track time
computation_time = 0
n_operations = 0
unrecognized_matched_faces = 0
unrecognized_mismatched_faces = 0


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
                start_time__computation_time = time.time()
                face, face_prob = mtcnn(images[i], return_prob=True)
                if face is not None and face_prob > 0.00:
                    emb = resnet(face.unsqueeze(0))
                    embeddings.append(emb.detach())
                    elapsed_time_mismatched_computation_time = time.time() - start_time__computation_time
                    computation_time += elapsed_time_mismatched_computation_time
                    n_operations += 1
                else:
                    print(f"No face detected in {os.path.basename(image_path)}")
                    unrecognized_matched_faces += 1
                    break
            if len(embeddings) == 2:
                matched_embedding_list.extend(embeddings)
                matched_name_list.extend([folder] * 2)
            else:
                print(f"Not enough faces detected in {folder_path}")

# generate embeddings for the dataset
mismatched_embedding_list = []
mismatched_name_list = []
processed_indices = set()
max_images = 3360

for image, index in mismatched_loader:
    if index >= max_images:
        break
    start_time__computation_time = time.time()
    face, face_prob = mtcnn(image, return_prob=True)
    if face is not None and face_prob > 0.00:
        emb = resnet(face.unsqueeze(0))
        mismatched_embedding_list.append(emb.detach())
        mismatched_name_list.append(mismatched_dataset.idx_to_class[index])
        elapsed_time_mismatched_computation_time = time.time() - start_time__computation_time
        computation_time += elapsed_time_mismatched_computation_time
        n_operations += 1
    else:
        print("No face detected in image:", index)
        unrecognized_mismatched_faces += 1

if len(mismatched_embedding_list) % 2 != 0:
    mismatched_embedding_list.pop()
    mismatched_name_list.pop()


dist_matched = [torch.dist(matched_embedding_list[i], matched_embedding_list[i + 1]).item() for i in range(0, len(matched_embedding_list), 2)]
dist_mismatched = [torch.dist(mismatched_embedding_list[i], mismatched_embedding_list[i + 1]).item() for i in range(0, len(mismatched_embedding_list), 2)]

# Calculate average distances
avg_dist_matched = np.mean(dist_matched)
avg_dist_mismatched = np.mean(dist_mismatched)

distances = dist_matched + dist_mismatched
labels = [1] * len(dist_matched) + [0] * len(dist_mismatched)


accuracies = []
thresholds = np.linspace(0.1, 1.6, 5000)
FARs = []
FRRs = []
for threshold in thresholds:
    FAR, FRR = calculate_far_frr(labels, distances, threshold)
    accuracy = calculate_accuracy(labels, distances, threshold)
    FARs.append(FAR)
    FRRs.append(FRR)
    accuracies.append(accuracy)

max_accuracy_index = np.argmax(accuracies)
max_accuracy_threshold = thresholds[max_accuracy_index]
eer_index = np.argmin(np.abs(np.array(FARs) - np.array(FRRs)))
eer = (FARs[eer_index] + FRRs[eer_index]) / 2

# Calculate the accuracy and F1 score of the model
true_matched = []
true_mismatched = []
pred_matched = []
pred_mismatched = []

# Compare embeddings and calculate metrics for matched faces
for i in range(0, len(matched_embedding_list), 2):
    emb1 = matched_embedding_list[i]
    emb2 = matched_embedding_list[i + 1]
    dist = torch.dist(emb1, emb2).item()
    is_match = matched_name_list[i] == matched_name_list[i + 1]
    true_matched.append(is_match)
    pred_matched.append(dist < 1.062)

# Compare embeddings and calculate metrics for mismatched faces
for i in range(0, len(mismatched_embedding_list), 2):
    emb1 = mismatched_embedding_list[i]
    emb2 = mismatched_embedding_list[i + 1]
    dist = torch.dist(emb1, emb2).item()
    is_mismatch = mismatched_name_list[i] != mismatched_name_list[i + 1]
    true_mismatched.append(is_mismatch)
    pred_mismatched.append(dist > 1.062)

# Generate random predictions for unrecognized faces
random_matched_predictions = [random.choice([True, False]) for _ in range(unrecognized_matched_faces)]
random_mismatched_predictions = [random.choice([True, False]) for _ in range(unrecognized_mismatched_faces // 2)]

# Update the counts for recognized faces
True_Positives = sum([same_face and pred for same_face, pred in zip(true_matched, pred_matched)]) # True Positives
True_Negatives = sum([different_face and pred for different_face, pred in zip(true_mismatched, pred_mismatched)]) # True Negatives
False_Negatives = sum([same_face and not pred for same_face, pred in zip(true_matched, pred_matched)]) # False Negatives
False_Positives = sum([different_face and not pred for different_face, pred in zip(true_mismatched, pred_mismatched)]) # False Positives

# Include random predictions for unrecognized faces
True_Positives += sum(random_matched_predictions)
False_Negatives += len(random_matched_predictions) - sum(random_matched_predictions)
True_Negatives += sum(random_mismatched_predictions)
False_Positives += len(random_mismatched_predictions) - sum(random_mismatched_predictions)

# Calculate detection rate
total_images = len(matched_embedding_list) + len(mismatched_embedding_list) + unrecognized_matched_faces + unrecognized_mismatched_faces
detected_faces = total_images - unrecognized_matched_faces - unrecognized_mismatched_faces
detection_percentage = 100 * (detected_faces / total_images)

# Calculate total predictions
total_predictions = True_Positives + False_Negatives + True_Negatives + False_Positives

# Calculate accuracy, precision, recall, and F1 score
accuracy = (True_Positives + True_Negatives) / total_predictions
precision = True_Positives / (True_Positives + False_Positives)
recall = True_Positives / (True_Positives + False_Negatives)
f1 = 2 * precision * recall / (precision + recall)
# Calculate the average InceptionResnetV1 time
average_computation_time = computation_time / n_operations

print("Max Accuracy Threshold:", max_accuracy_threshold)
print("Average Computation time:", average_computation_time)
print("Total images not recognised in matched_faces:", unrecognized_matched_faces)
print("Total images not recognised in mismatched_faces:", unrecognized_mismatched_faces)
print("Detection Rate:", detection_percentage)
print("True Positives: ", True_Positives)
print("True Negatives: ", True_Negatives)
print("False Negatives: ", False_Negatives)
print("False Positives: ", False_Positives)
print("Average distance for matched faces:", avg_dist_matched)
print("Average distance for mismatched faces:", avg_dist_mismatched)
print("EER:", eer)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)


# Scatter plot
fig, ax = plt.subplots()

def add_jitter(values, jitter_amount):
    return [value + jitter_amount * (2 * np.random.rand() - 1) for value in values]

# Increase jitter amount
jitter_amount = 0.45

# Define plot title
plot_title = "MTCNN with InceptionresnetV1 Pre-trained on Casia-Webface tested on the LFW Dataset- 20 Percent Square Size Occlusion"

dist_true_positives = [dist for dist, same_face, pred in zip(dist_matched, true_matched, pred_matched) if same_face == True and pred == True]
dist_true_negatives = [dist for dist, different_face, pred in zip(dist_mismatched, true_mismatched, pred_mismatched) if different_face == True and pred == True]
dist_false_negatives = [dist for dist, same_face, pred in zip(dist_matched, true_matched, pred_matched) if same_face == True and pred == False]
dist_false_positives = [dist for dist, different_face, pred in zip(dist_mismatched, true_mismatched, pred_mismatched) if different_face == True and pred == False]

# True positives
ax.errorbar(dist_true_positives, add_jitter([0] * len(dist_true_positives), jitter_amount), fmt='s', c='green', alpha=0.5, label='True Positives')
# False negatives
ax.errorbar(dist_false_negatives, add_jitter([0] * len(dist_false_negatives), jitter_amount), fmt='p', c='red', alpha=0.5, label='False Negatives')
# True negatives
ax.errorbar(dist_true_negatives, add_jitter([1] * len(dist_true_negatives), jitter_amount), fmt='o', c='blue', alpha=0.5, label='True Negatives')
# False positives
ax.errorbar(dist_false_positives, add_jitter([1] * len(dist_false_positives), jitter_amount), fmt='P', c='orange', alpha=0.5, label='False Positives')

# EER threshold
ax.axvline(x=1.062, color='purple', linestyle='-', label='Threshold Distance')
ax.axvline(x=avg_dist_matched, color='black', linestyle='-.', label='Average Matched Distance')
ax.axvline(x=avg_dist_mismatched, color='black', linestyle='--', label='Average Mismatched Distance')

ax.set_xlabel('Distance')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Matched Faces','Mismatched Faces'])
ax.set_title(plot_title)
ax.legend()

plt.show()