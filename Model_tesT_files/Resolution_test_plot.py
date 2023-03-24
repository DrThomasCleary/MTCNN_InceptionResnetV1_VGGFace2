# import matplotlib.pyplot as plt

# # Define the metrics for each resolution level
# resolution = ['Original-112', '128', '96', '64', '48', '32']
# eer = [0.0780736409608091, 0.07027145666526197, 0.05827170992050815, 0.07640887493566309, 0.12992727516661998, 0.23372781065088757]
# accuracy = [0.9324178782983307, 0.9369951534733441, 0.9429186860527733, 0.7831941826016698, 0.5499596014004848, 0.09453272286560732]
# precision = [0.8892857142857142, 0.9035714285714286, 0.9290822407628129, 0.7751889168765743, 0.50033760972316, 0.010227272727272727]
# recall = [0.9583066067992303, 0.9547169811320755, 0.9437046004842615, 0.7331745086360929, 0.4431818181818182, 0.01098901098901099]
# f1_score = [0.9225069465884532, 0.9284403669724771, 0.9363363363363365, 0.7535965717783899, 0.47002854424357754, 0.010594467333725722]
# detection_rate = [1.0, 1.0, 0.9989224137931034, 0.8854122621564483, 0.6992458521870286, 0.13867859600825877]  # Normalized detection rates

# # Create the plot
# plt.plot(resolution, eer, marker='o', label='EER')
# plt.plot(resolution, accuracy, marker='s', label='Accuracy')
# plt.plot(resolution, precision, marker='^', label='Precision')
# plt.plot(resolution, recall, marker='d', label='Recall')
# plt.plot(resolution, f1_score, marker='*', label='F1 score')
# plt.plot(resolution, detection_rate, marker='x', label='Detection Rate (Normalized)')

# # Set the x-axis label and tick marks
# plt.xlabel('Resolution')
# plt.xticks(resolution)

# # Set the y-axis label and tick marks
# plt.ylabel('Metrics')
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# # Add a legend to the plot
# plt.legend()

# # Display the plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

def color_map(percentage_change, reverse=False):
    if reverse:
        return 'green' if percentage_change <= 0 else 'red'
    else:
        return 'green' if percentage_change >= 0 else 'red'

# define the data
resolution = ['128x128', '96x96', '64x64', '48x48', '32x32']
accuracy = [93.70, 94.29, 78.32, 55.00, 9.45]
precision = [90.36, 92.91, 77.52, 50.03, 1.02]
recall = [95.47, 94.37, 73.32, 44.32, 1.10]
f1_score = [92.84, 93.63, 75.36, 47.00, 1.06]
eer = [7.03, 5.83, 7.64, 12.99, 23.37]
inception_time = [0.0648, 0.0617, 0.0570, 0.0570, 0.0571]
detection_rate = [100.0, 99.89, 88.54, 69.92, 13.87]

# original data
original_accuracy = 93.24
original_precision = 88.93
original_recall = 95.83
original_f1_score = 92.25
original_eer = 7.81
original_inception_time = 0.0767
original_detection_rate = 100.0

# calculate the percentage changes
accuracy_change = [(acc - original_accuracy) / original_accuracy * 100 for acc in accuracy]
precision_change = [(pre - original_precision) / original_precision * 100 for pre in precision]
recall_change = [(rec - original_recall) / original_recall * 100 for rec in recall]
f1_score_change = [(f1 - original_f1_score) / original_f1_score * 100 for f1 in f1_score]
eer_change = [(e - original_eer) / original_eer * 100 for e in eer]
inception_time_change = [(t - original_inception_time) / original_inception_time * 100 for t in inception_time]
detection_rate_change = [(dr - original_detection_rate) / original_detection_rate * 100 for dr in detection_rate]


# # set the bar width
# bar_width = 0.1

# # create the plot
# fig, ax = plt.subplots()
# for i, (data, color, label) in enumerate(zip([accuracy_change, precision_change, recall_change, f1_score_change, eer_change, inception_time_change, detection_rate_change],
#                                               ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'cyan'],
#                                               ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate'])):
#     ax.bar(np.arange(len(resolution)) + i * bar_width, data, width=bar_width, color=color, label=label)

# # add x-axis labels and title
# ax.set_xlabel('Resolution')
# ax.set_ylabel('Percentage Change')
# ax.set_xticks(np.arange(len(resolution)) + bar_width * 3)
# ax.set_xticklabels(resolution)
# plt.xticks(rotation=45, ha='right')
# plt.subplots_adjust(bottom=0.2)
# plt.title('Performance Metrics by Resolution')

# # add a legend
# plt.legend()

# # display the plot
# plt.show()

# below is the code for the grouped bar chat
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

metrics = [accuracy_change, precision_change, recall_change, f1_score_change, eer_change, inception_time_change, detection_rate_change]

# Create a list of metrics and their corresponding names
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate']
reverse = [False, False, False, False, True, True, False]

axs = axs.flat  # Flatten the axs array for easier iteration

for i, (data, label, rev, ax) in enumerate(zip(metrics, labels, reverse, axs)):
    rects = ax.bar(resolution, data, color=[color_map(x, reverse=rev) for x in data])
    ax.set_title(label)
    ax.set_xticks(range(len(resolution)))  # Set the tick positions
    ax.set_xticklabels(resolution, rotation=25, ha='right')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Percentage Change')
    ax.set_xlim(-0.5, len(resolution) - 0.5)  # Set the x-axis limits

# Remove the last unused subplot
fig.delaxes(axs[7])

fig.suptitle('Performance Metrics by resolution level', fontsize=16)
plt.show()

