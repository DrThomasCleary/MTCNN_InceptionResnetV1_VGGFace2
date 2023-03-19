import matplotlib.pyplot as plt

# Define the metrics for each intensity level
intensity = [1, 2, 3, 4, 5]
eer = [0.1486, 0.1699, 0.2036, 0.2691, 0.3290]
accuracy = [0.8514, 0.8301, 0.7964, 0.7309, 0.6711]
precision = [0.8512, 0.8304, 0.7970, 0.7309, 0.6699]
recall = [0.8256, 0.8013, 0.7634, 0.6917, 0.6262]
f1_score = [0.8382, 0.8156, 0.7798, 0.7108, 0.6473]

# Create the plot
plt.plot(intensity, eer, marker='o', label='EER')
plt.plot(intensity, accuracy, marker='s', label='Accuracy')
plt.plot(intensity, precision, marker='^', label='Precision')
plt.plot(intensity, recall, marker='d', label='Recall')
plt.plot(intensity, f1_score, marker='*', label='F1 score')


# Set the x-axis label and tick marks
plt.xlabel('Blurry Test Intensity')
plt.xticks(intensity)

# Set the y-axis label and tick marks
plt.ylabel('Metrics')
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# define the data
intensity = ['Intensity 0', 'Intensity 1', 'Intensity 2', 'Intensity 3', 'Intensity 4', 'Intensity 5']
accuracy = [86.05, 85.14, 83.01, 79.64, 73.09, 67.11]
precision = [86.13, 85.12, 83.04, 79.70, 73.09, 66.99]
recall = [83.54, 82.56, 80.13, 76.34, 69.17, 62.62]
f1_score = [84.82, 83.82, 81.56, 77.98, 71.08, 64.73]
eer = [13.94, 14.86, 16.99, 20.36, 26.91, 32.90]

# calculate the percentage changes
accuracy_change = [(accuracy[i]-accuracy[0])/accuracy[0]*100 for i in range(len(accuracy))]
precision_change = [(precision[i]-precision[0])/precision[0]*100 for i in range(len(precision))]
recall_change = [(recall[i]-recall[0])/recall[0]*100 for i in range(len(recall))]
f1_score_change = [(f1_score[i]-f1_score[0])/f1_score[0]*100 for i in range(len(f1_score))]
eer_change = [(eer[i]-eer[0])/eer[0]*100 for i in range(len(eer))]

# set the bar width
bar_width = 0.15

# create the plot
fig, ax = plt.subplots()
for i, (data, color, label) in enumerate(zip([accuracy_change, precision_change, recall_change, f1_score_change, eer_change],
                                              ['green', 'red', 'blue', 'orange', 'purple'],
                                              ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER'])):
    ax.bar(np.arange(len(intensity)) + i * bar_width, data, width=bar_width, color=color, label=label)

# add x-axis labels and title
ax.set_xlabel('Blurry Intensity')
ax.set_ylabel('Percentage Change')
ax.set_xticks(np.arange(len(intensity)) + bar_width * 2)
ax.set_xticklabels(intensity)
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2)
plt.title('Performance Metrics by Blurry Intensity')

# add a legend
plt.legend()

# display the plot
plt.show()
