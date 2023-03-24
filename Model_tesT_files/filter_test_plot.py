import matplotlib.pyplot as plt
import numpy as np

def percentage_change(original, new):
    return (new - original) / original * 100

# Function to convert the percentage change to colors
def color_map(value, reverse=False):
    if reverse:
        value = -value
    if value < 0:
        return 'red'
    else:
        return 'green'
    
filters = ['Sepia', 'Grayscale', 'Brightness x1.5', 'Colour Tint', 'Contrast 2x']
original_eer = 0.0780736409608091
original_avg_time = 0.0767136300258503
original_detection_rate = 100.0
original_accuracy = 0.9324178782983307
original_precision = 0.8892857142857142
original_recall = 0.9583066067992303
original_f1_score = 0.9225069465884532

eers = [0.07306809756646926, 0.06747169032324846, 0.08349075359944926, 0.06924035866966627, 0.09084663897535047]
avg_times = [0.07199593831098171, 0.06941452956498755, 0.07442977320900877, 0.06524384171728785, 0.07333637278760953]
detection_rates = [99.93265993265993, 99.82486865148861, 99.54128440366972, 96.8647764449291, 99.71694298422969]
accuracies = [0.9329383248047401, 0.9394021007271748, 0.9243199569081605, 0.8949919224555735, 0.9189553042541734]
precisions = [0.9064919594997022, 0.9107142857142857, 0.8805256869772998, 0.8672299336149668, 0.8747763864042933]
recalls = [0.942998760842627, 0.9532710280373832, 0.9479099678456592, 0.8942128189172371, 0.941591784338896]
f1_scores = [0.9243850592165198, 0.9315068493150684, 0.9129761536079282, 0.880514705882353, 0.9069551777434312]

eers_pct_change = [percentage_change(original_eer, x) for x in eers]
avg_times_pct_change = [percentage_change(original_avg_time, x) for x in avg_times]
detection_rates_pct_change = [percentage_change(original_detection_rate, x) for x in detection_rates]
accuracies_pct_change = [percentage_change(original_accuracy, x) for x in accuracies]
precisions_pct_change = [percentage_change(original_precision, x) for x in precisions]
recalls_pct_change = [percentage_change(original_recall, x) for x in recalls]
f1_scores_pct_change = [percentage_change(original_f1_score, x) for x in f1_scores]

# x = np.arange(len(filters))
# width = 0.12

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 2*width, eers_pct_change, width, label='EER')
# rects2 = ax.bar(x - width, avg_times_pct_change, width, label='Average InceptionResnetV1 Time')
# rects3 = ax.bar(x, accuracies_pct_change, width, label='Accuracy')
# rects4 = ax.bar(x + width, precisions_pct_change, width, label='Precision')
# rects5 = ax.bar(x + 2*width, recalls_pct_change, width, label='Recall')
# rects6 = ax.bar(x + 3*width, f1_scores_pct_change, width, label='F1 score')
# rects7 = ax.bar(x + 4*width, detection_rates_pct_change, width, label='Detection Rate')

# ax.set_ylabel('Percentage Change')
# ax.set_title('Percentage Change of Metrics by Filter')
# ax.set_xticks(x)
# ax.set_xticklabels(filters)
# ax.legend()

# fig.tight_layout()
# plt.show()

# Set the filters as the x-axis labels
intensity = filters

# Create a list of metrics and their corresponding names
metrics = [accuracies_pct_change, precisions_pct_change, recalls_pct_change, f1_scores_pct_change, eers_pct_change, avg_times_pct_change, detection_rates_pct_change]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate']
reverse = [False, False, False, False, True, True, False]

# Create the grouped bar chart
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

axs = axs.flat  # Flatten the axs array for easier iteration

for i, (data, label, rev, ax) in enumerate(zip(metrics, labels, reverse, axs)):
    rects = ax.bar(intensity, data, color=[color_map(x, reverse=rev) for x in data])
    ax.set_title(label)
    ax.set_xticks(range(len(intensity)))  # Set the tick positions
    ax.set_xticklabels(intensity, rotation=25, ha='right')
    ax.set_xlabel('Filter')
    ax.set_ylabel('Percentage Change')
    ax.set_xlim(-0.5, len(intensity) - 0.5)  # Set the x-axis limits

# Remove the last unused subplot
fig.delaxes(axs[7])

fig.suptitle('Performance Metrics by Filter', fontsize=16)
plt.show()