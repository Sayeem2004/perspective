import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
acronym_toxic_path = '../classification/acronym/acronym-toxic_TOXICITY.csv'
non_acronym_toxic_path = '../classification/acronym/no-acronym-toxic_TOXICITY.csv'
acronym_nice_path = '../classification/acronym/acronym-nice_TOXICITY.csv'
non_acronym_nice_path = '../classification/acronym/no-acronym-nice_TOXICITY.csv'

# Load each dataset and compute the average of the third column (score)
def get_avg_score(path):
    count = 0
    with open(f'{path}', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # find the last comma
            last_comma = line.rfind(',')
            # get the last part of the line
            last_part = line[last_comma+1:]
            count += float(last_part)
    return count / len(lines)

# Compute averages
avg_scores = {
    'Acronym Toxic': get_avg_score(acronym_toxic_path),
    'Non-Acronym Toxic': get_avg_score(non_acronym_toxic_path),
    'Acronym Nice': get_avg_score(acronym_nice_path),
    'Non-Acronym Nice': get_avg_score(non_acronym_nice_path),
}

# Reorganize: group by toxicity
acronym_scores = [avg_scores['Acronym Toxic'], avg_scores['Acronym Nice']]
non_acronym_scores = [avg_scores['Non-Acronym Toxic'], avg_scores['Non-Acronym Nice']]

# Plotting
labels = ['Toxic', 'Nice']  # New x-axis: toxicity groups
x = np.arange(len(labels))  # [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x + width/2, acronym_scores, width, label='Acronym', color='#1f77b4')
bar2 = ax.bar(x - width/2, non_acronym_scores, width, label='Non-Acronym', color='#ff7f0e')

# Add value labels
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16)

# Labels and layout
ax.set_ylabel('Average Toxicity Score', fontsize=24)
ax.set_title('Toxicity Scores by Comment Type', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24)
ax.legend(fontsize=24)

plt.ylim(0, max(acronym_scores + non_acronym_scores) + 0.05)
plt.tight_layout()
plt.savefig('acronym_graph.png', dpi=300)
plt.show()
