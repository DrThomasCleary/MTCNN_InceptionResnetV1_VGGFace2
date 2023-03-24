# import matplotlib.pyplot as plt

# # Define the metrics for each intensity level
# intensity = [0, 1, 2, 3, 4, 5]
# eer = [0.0780736409608091, 0.08348106021341775, 0.09103497645984118, 0.11918382488463033, 0.1616011106487995, 0.237210278391713]
# accuracy = [0.9324178782983307, 0.9286483575659666, 0.9219170705438879, 0.8920010772959871, 0.8446000538647993, 0.7632642068408295]
# precision = [0.8892857142857142, 0.8892197736748064, 0.8738845925044616, 0.8484486873508353, 0.7790351399642644, 0.6798561151079137]
# recall = [0.9583066067992303, 0.9497455470737913, 0.9495798319327731, 0.9063097514340345, 0.8639365918097754, 0.7667342799188641]
# f1_score = [0.9225069465884532, 0.9184866195016917, 0.9101610904584881, 0.8764252696456087, 0.8192922016911997, 0.7206863679694948]
# detection_rate = [1, 0.9998653561330282, 0.9997307485191168, 0.9993264178903408, 0.9973045822102425, 0.9948690251147718]

# # Create the plot
# plt.plot(intensity, eer, marker='o', label='EER')
# plt.plot(intensity, accuracy, marker='s', label='Accuracy')
# plt.plot(intensity, precision, marker='^', label='Precision')
# plt.plot(intensity, recall, marker='d', label='Recall')
# plt.plot(intensity, f1_score, marker='*', label='F1 score')
# plt.plot(intensity, detection_rate, marker='p', label='Detection Rate')

# # Set the x-axis label and tick marks
# plt.xlabel('Blurry Test Intensity')
# plt.xticks(intensity)

# # Set the y-axis label and tick marks
# plt.ylabel('Metrics')
# plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

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
intensity = ['Intensity 1', 'Intensity 2', 'Intensity 3', 'Intensity 4', 'Intensity 5']
inception_time = [ 0.07639169089401825,0.07651712218288974,0.07925893071043705,0.07776997428511698,0.07299592839257192]
detection_rate = [99.98653561330282, 99.97307485191168, 99.93264178903408, 99.73045822102425, 99.48690251147718]
accuracy = [92.86483575659666, 92.19170705438879, 89.20010772959871, 84.46000538647993, 76.32642068408295]
precision = [88.92197736748064, 87.38845925044616, 84.84486873508353, 77.90351399642644, 67.98561151079137]
recall = [94.97455470737913, 94.95798319327731, 90.63097514340345, 86.39365918097754, 76.67342799188641]
f1_score = [91.84866195016917, 91.01610904584881, 87.64252699656087, 81.92922016911997, 72.06863679694948]
eer = [8.348106021341775, 9.103497645984118, 11.918382488463033, 16.16011106487995, 23.7210278391713]

# define the data for Intensity 0
accuracy_0 = 93.24
precision_0 = 88.93
recall_0 = 95.83
f1_score_0 = 92.25
eer_0 = 7.81
detection_rate_0 = 100
original_inception_time = 0.0767136300258503

# calculate the percentage changes
accuracy_change = [(accuracy[i] - accuracy_0) / accuracy_0 * 100 for i in range(len(accuracy))]
precision_change = [(precision[i] - precision_0) / precision_0 * 100 for i in range(len(precision))]
recall_change = [(recall[i] - recall_0) / recall_0 * 100 for i in range(len(recall))]
f1_score_change = [(f1_score[i] - f1_score_0) / f1_score_0 * 100 for i in range(len(f1_score))]
eer_change = [(eer[i] - eer_0) / eer_0 * 100 for i in range(len(eer))]
detection_rate_change = [(detection_rate[i] - detection_rate_0) / detection_rate_0 * 100 for i in range(len(detection_rate))]
inception_time_change = [(t - original_inception_time) / original_inception_time * 100 for t in inception_time]

# # set the bar width
# bar_width = 0.15

# fig, ax = plt.subplots()
# for i, (data, color, label) in enumerate(zip([accuracy_change, precision_change, recall_change, f1_score_change, eer_change, inception_time_change, detection_rate_change],
#                                              ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'cyan'],
#                                              ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate'])):
#     ax.bar(np.arange(len(intensity)) + i * bar_width, data, width=bar_width, color=color, label=label)

# # Update x-axis labels and title
# ax.set_xlabel('Blurry Intensity')
# ax.set_ylabel('Percentage Change')
# ax.set_xticks(np.arange(len(intensity)) + bar_width * 3)
# ax.set_xticklabels(intensity)
# plt.xticks(rotation=45, ha='right')
# plt.subplots_adjust(bottom=0.2)
# plt.title('Performance Metrics by Blurry Intensity')

# # Add a legend
# plt.legend()

# # Display the plot
# plt.show()

#  below is for the grouped bar chart 
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

metrics = [accuracy_change, precision_change, recall_change, f1_score_change, eer_change, inception_time_change, detection_rate_change]

# Create a list of metrics and their corresponding names
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate']
reverse = [False, False, False, False, True, True, False]

axs = axs.flat  # Flatten the axs array for easier iteration

for i, (data, label, rev, ax) in enumerate(zip(metrics, labels, reverse, axs)):
    rects = ax.bar(intensity, data, color=[color_map(x, reverse=rev) for x in data])
    ax.set_title(label)
    ax.set_xticks(range(len(intensity)))  # Set the tick positions
    ax.set_xticklabels(intensity, rotation=25, ha='right')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Percentage Change')
    ax.set_xlim(-0.5, len(intensity) - 0.5)  # Set the x-axis limits

# Remove the last unused subplot
fig.delaxes(axs[7])

fig.suptitle('Performance Metrics by intensity level', fontsize=16)
plt.show()

