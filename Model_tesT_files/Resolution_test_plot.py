import matplotlib.pyplot as plt


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

