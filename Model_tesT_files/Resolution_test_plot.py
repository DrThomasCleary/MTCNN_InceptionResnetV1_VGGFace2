import matplotlib.pyplot as plt

# Define the metrics for each resolution level
resolution = [256, 128, 64, 32]
eer = [0.13940423514538558, 0.1400512712459615, 0.17876457367607812, 0.3493849761137512]
accuracy = [0.8605277329025309, 0.8599892299407647, 0.8212170166935918, 0.6508674101610905]
precision = [0.8613095238095239, 0.8595238095238096, 0.8214285714285714, 0.6488439306358381]
recall = [0.8354503464203233, 0.8356481481481481, 0.7912844036697247, 0.5834957764782326]
f1_score = [0.8481828839390387, 0.8474178403755869, 0.8060747663551402, 0.6144372220321587]

# Create the plot
plt.plot(resolution, eer, marker='o', label='EER')
plt.plot(resolution, accuracy, marker='s', label='Accuracy')
plt.plot(resolution, precision, marker='^', label='Precision')
plt.plot(resolution, recall, marker='d', label='Recall')
plt.plot(resolution, f1_score, marker='*', label='F1 score')


# Set the x-axis label and tick marks
plt.xlabel('Resolution')
plt.xticks(resolution)

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
resolution = ['256x256', '128x128', '64x64', '32x32']
accuracy = [86.05, 85.99, 82.12, 65.09]
precision = [86.13, 85.95, 82.14, 64.88]
recall = [83.54, 83.56, 79.13, 58.35]
f1_score = [84.82, 84.74, 80.61, 61.44]
eer = [13.94, 14.01, 17.88, 34.94]

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
    ax.bar(np.arange(len(resolution)) + i * bar_width, data, width=bar_width, color=color, label=label)

# add x-axis labels and title
ax.set_xlabel('Resolution')
ax.set_ylabel('Percentage Change')
ax.set_xticks(np.arange(len(resolution)) + bar_width * 2)
ax.set_xticklabels(resolution)
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2)
plt.title('Performance Metrics by Resolution')

# add a legend
plt.legend()

# display the plot
plt.show()