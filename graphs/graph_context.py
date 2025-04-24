import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('../classification/context/mandarin_TOXICITY.csv', header=None)
df.columns = ['text', 'label', 'score']
df['label'] = df['label'].astype(str).str.strip().str.upper()
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['length'] = df['text'].str.len()

# Assign language types: alternating Chinese, Mixed, English
language_map = {0: 'Chinese', 1: 'Mixed', 2: 'English'}
df['language_type'] = df.index % 3
df['language_type'] = df['language_type'].map(language_map)

# Classify into TP, FP, TN, FN
def classify(label, score):
    if label == 'T' and score >= 0.5:
        return 'TP'
    elif label == 'T' and score < 0.5:
        return 'FN'
    elif label == 'F' and score < 0.5:
        return 'TN'
    elif label == 'F' and score >= 0.5:
        return 'FP'
    else:
        return None

df['quad_class'] = df.apply(lambda row: classify(row['label'], row['score']), axis=1)
df = df[df['quad_class'].notnull()]

# Color map
color_map = {'TP': 'green', 'FP': 'red', 'TN': 'blue', 'FN': 'orange'}
languages = ['Chinese', 'Mixed', 'English']

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Toxicity Detection by Text Length and Score – Split by Language', fontsize=16)

for i, lang in enumerate(languages):
    ax = axes[i]
    lang_df = df[df['language_type'] == lang]
    for cls, group in lang_df.groupby('quad_class'):
        ax.scatter(group['length'], group['score'], c=color_map[cls], label=cls, alpha=0.8)

    # Axis and lines
    median_len = lang_df['length'].median()
    ax.axhline(0.5, color='black', linewidth=1.5)
    ax.axvline(median_len, color='black', linewidth=1.5)
    ax.set_xlim(0, lang_df['length'].max() + 10)
    ax.set_ylim(0, 1)
    ax.set_title(lang)
    ax.set_xlabel('Text Length (Characters)')
    if i == 0:
        ax.set_ylabel('Toxicity Score')
    else:
        ax.set_yticklabels([])

    # Dynamic label positioning
    x_left = median_len - (lang_df['length'].max() * 0.3)
    x_right = median_len + (lang_df['length'].max() * 0.05)
    y_top = 0.85
    y_bottom = 0.1

    ax.text(x_right, y_top, 'TP', color='green', fontsize=10, weight='bold')
    ax.text(x_left, y_top, 'FP', color='red', fontsize=10, weight='bold')
    ax.text(x_left, y_bottom, 'TN', color='blue', fontsize=10, weight='bold')
    ax.text(x_right, y_bottom, 'FN', color='orange', fontsize=10, weight='bold')

    ax.grid(True, linestyle='--', alpha=0.6)

    # Add legend to first subplot only
    if i == 0:
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='TP', markerfacecolor='green', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='FP', markerfacecolor='red', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='TN', markerfacecolor='blue', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='FN', markerfacecolor='orange', markersize=8)
        ], title='Classification', loc='upper right', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('toxicity_quadrants_by_language_fixed.png', dpi=300)
plt.show()
