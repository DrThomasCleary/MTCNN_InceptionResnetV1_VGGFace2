import matplotlib.pyplot as plt
import numpy as np

def percentage_change(original, new):
    return (new - original) / original * 100

def color_map(percentage_change, reverse=False):
    if reverse:
        return 'green' if percentage_change <= 0 else 'red'
    else:
        return 'green' if percentage_change >= 0 else 'red'
    
occlusion_sizes = ['5%', '10%', '15%', '20%']
x = np.arange(len(occlusion_sizes))
width = 0.15

fig, ax = plt.subplots()

original_eer = 0.0780736409608091
original_accuracy = 0.9324178782983307
original_precision = 0.8892857142857142
original_recall = 0.9583066067992303
original_f1_score = 0.9225069465884532
original_inception_time = 0.0767136300258503
original_detection_rate = 100.0

eers = [0.11046930238584413, 0.14720267010343346, 0.22837519931137437, 0.3893006076492315]
accuracies = [0.8879913839526118, 0.7654819601507808, 0.4501884760366182, 0.15270670616751952]
precisions = [0.8584392014519057, 0.6916771752837326, 0.31163434903047094, 0.023751522533495738]
recalls = [0.8863210493441599, 0.7417173766058147, 0.30040053404539385, 0.02465233881163085]
f1_scores = [0.8721573448063922, 0.7158238172920065, 0.30591434398368456, 0.024193548387096774]
inception_times = [0.07620683238688507, 0.07348971840672136, 0.07136492017127825, 0.13074201307257874]
detection_rates = [98.28990228013029, 90.27193720213063, 61.77701969297566, 23.69]

eers_pct_change = [percentage_change(original_eer, x) for x in eers]
accuracies_pct_change = [percentage_change(original_accuracy, x) for x in accuracies]
precisions_pct_change = [percentage_change(original_precision, x) for x in precisions]
recalls_pct_change = [percentage_change(original_recall, x) for x in recalls]
f1_scores_pct_change = [percentage_change(original_f1_score, x) for x in f1_scores]
inception_times_pct_change = [percentage_change(original_inception_time, x) for x in inception_times]
detection_rates_pct_change = [percentage_change(original_detection_rate, x) for x in detection_rates]

# # set the bar width
# bar_width = 0.1

# # create the plot
# fig, ax = plt.subplots()
# for i, (data, color, label) in enumerate(zip([accuracies_pct_change, precisions_pct_change, recalls_pct_change, f1_scores_pct_change, eers_pct_change, inception_times_pct_change, detection_rates_pct_change],
#                                               ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'cyan'],
#                                               ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate'])):
#     ax.bar(np.arange(len(occlusion_sizes)) + i * bar_width, data, width=bar_width, color=color, label=label)

# # add x-axis labels and title
# ax.set_xlabel('Occlusion Square Size')
# ax.set_ylabel('Percentage Change')
# ax.set_xticks(np.arange(len(occlusion_sizes)) + bar_width * 3)
# ax.set_xticklabels(occlusion_sizes)
# plt.xticks(rotation=45, ha='right')
# plt.subplots_adjust(bottom=0.2)
# plt.title('Performance Metrics by Occlusion Square Size')

# # add a legend
# plt.legend()

# # display the plot
# plt.show()



fig, axs = plt.subplots(2, 4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

metrics = [accuracies_pct_change, precisions_pct_change, recalls_pct_change, f1_scores_pct_change, eers_pct_change, inception_times_pct_change, detection_rates_pct_change]

# Create a list of metrics and their corresponding names
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EER', 'Inception Time', 'Detection Rate']
reverse = [False, False, False, False, True, True, False]

axs = axs.flat  # Flatten the axs array for easier iteration

for i, (data, label, rev, ax) in enumerate(zip(metrics, labels, reverse, axs)):
    rects = ax.bar(occlusion_sizes, data, color=[color_map(x, reverse=rev) for x in data])
    ax.set_title(label)
    ax.set_xticks(range(len(occlusion_sizes)))  # Set the tick positions
    ax.set_xticklabels(occlusion_sizes, rotation=25, ha='right')
    ax.set_xlabel('Square size')
    ax.set_ylabel('Percentage Change')
    ax.set_xlim(-0.5, len(occlusion_sizes) - 0.5)  # Set the x-axis limits

# Remove the last unused subplot
fig.delaxes(axs[7])

fig.suptitle('Performance Metrics by Occlusion square size', fontsize=16)
plt.show()