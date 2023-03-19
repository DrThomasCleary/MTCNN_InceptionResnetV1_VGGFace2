import matplotlib.pyplot as plt
import numpy as np

def percentage_change(original, new):
    return (new - original) / original * 100

filters = ['Sepia', 'Grayscale', '+50% Brightness', 'Colour tint', 'Contrast_2x']
original_eer_threshold = 1.045045045045045
original_eer = 0.13940423514538558
original_avg_time = 0.04951951002394184
original_accuracy = 0.8605277329025309
original_precision = 0.8613095238095239
original_recall = 0.8354503464203233
original_f1_score = 0.8481828839390387

eer_thresholds = [1.047047047047047, 1.027027027027027, 1.0610610610610611, 1.023023023023023, 1.0610610610610611]
eers = [0.14361930376630258, 0.13380196576843406, 0.15067598185519265, 0.15242418856098783, 0.1696990527306213]
avg_times = [0.050852007129098244, 0.04829106513553024, 0.0475918508011825, 0.0459661464773789, 0.0557527341286513]
accuracies = [0.8563729452977634, 0.8661268556005398, 0.8493224932249323, 0.8474957794034891, 0.8303161307754661]
precisions = [0.8564621798689697, 0.866945107398568, 0.8493397358943577, 0.8483322844556325, 0.8301435406698564]
recalls = [0.8312138728323699, 0.8418308227114716, 0.8226744186046512, 0.8174651303820497, 0.8013856812933026]
f1_scores = [0.8436491639777061, 0.8542034097589652, 0.8357944477259303, 0.8326127239036443, 0.8155111633372503]

eer_thresholds_pct_change = [percentage_change(original_eer_threshold, x) for x in eer_thresholds]
eers_pct_change = [percentage_change(original_eer, x) for x in eers]
avg_times_pct_change = [percentage_change(original_avg_time, x) for x in avg_times]
accuracies_pct_change = [percentage_change(original_accuracy, x) for x in accuracies]
precisions_pct_change = [percentage_change(original_precision, x) for x in precisions]
recalls_pct_change = [percentage_change(original_recall, x) for x in recalls]
f1_scores_pct_change = [percentage_change(original_f1_score, x) for x in f1_scores]

x = np.arange(len(filters))
width = 0.12

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3*width, eer_thresholds_pct_change, width, label='EER Threshold')
rects2 = ax.bar(x - 2*width, eers_pct_change, width, label='EER')
rects3 = ax.bar(x - width, avg_times_pct_change, width, label='Average InceptionResnetV1 Time')
rects4 = ax.bar(x, accuracies_pct_change, width, label='Accuracy')
rects5 = ax.bar(x + width, precisions_pct_change, width, label='Precision')
rects6 = ax.bar(x + 2*width, recalls_pct_change, width, label='Recall')
rects7 = ax.bar(x + 3*width, f1_scores_pct_change, width, label='F1 score')

ax.set_ylabel('Percentage Change')
ax.set_title('Percentage Change of Metrics by Filter')
ax.set_xticks(x)
ax.set_xticklabels(filters)
ax.legend()

fig.tight_layout()
plt.show()