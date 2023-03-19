import matplotlib.pyplot as plt
import numpy as np

def percentage_change(original, new):
    return (new - original) / original * 100

not_recognized_pct = [4, 14, 47, 84]
occlusion_sizes = ['5%', '10%', '15%', '20%']
x = np.arange(len(occlusion_sizes))
width = 0.15

fig, ax = plt.subplots()

occlusion_sizes = ['5%', '10%', '15%', '20%']
original_eer = 0.13940423514538558
original_accuracy = 0.8605277329025309
original_precision = 0.8613095238095239
original_recall = 0.8354503464203233
original_f1_score = 0.8481828839390387

eers = [0.20174213232538257, 0.2317982942982943, 0.2880380412137825, 0.4145958256457565]
accuracies = [0.7982938910291689, 0.768450184501845, 0.712039312039312, 0.5850746268656717]
precisions = [0.797911547911548, 0.7663817663817664, 0.7116991643454039, 0.5859375]
recalls = [0.7627715795654727, 0.7168554297135243, 0.5741573033707865, 0.25]
f1_scores = [0.7799459621735215, 0.7407917383821, 0.6355721393034826, 0.35046728971962615]

eers_pct_change = [percentage_change(original_eer, x) for x in eers]
accuracies_pct_change = [percentage_change(original_accuracy, x) for x in accuracies]
precisions_pct_change = [percentage_change(original_precision, x) for x in precisions]
recalls_pct_change = [percentage_change(original_recall, x) for x in recalls]
f1_scores_pct_change = [percentage_change(original_f1_score, x) for x in f1_scores]

x = np.arange(len(occlusion_sizes))
width = 0.15

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2 * width, eers_pct_change, width, label='EER')
rects2 = ax.bar(x - width, accuracies_pct_change, width, label='Accuracy')
rects3 = ax.bar(x, precisions_pct_change, width, label='Precision')
rects4 = ax.bar(x + width, recalls_pct_change, width, label='Recall')
rects5 = ax.bar(x + 2 * width, f1_scores_pct_change, width, label='F1 score')

ax.set_ylabel('Percentage Change')
ax.set_title('Percentage Change of Metrics by Occlusion Size')
ax.set_xticks(x)
ax.set_xticklabels(occlusion_sizes)
ax.legend(loc='upper left')

# Add the percentage of images not recognized as text annotations
annotation_y = 0.98  # Position for the first annotation
for i, pct in enumerate(not_recognized_pct):
    ax.annotate(
        f'{occlusion_sizes[i]} - {pct}% not recognized',
        xy=(1, annotation_y),
        xycoords='axes fraction',
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='top',
    )
    annotation_y -= 0.05  # Move the next annotation slightly lower

fig.tight_layout()
plt.show()