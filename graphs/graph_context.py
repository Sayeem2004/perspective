import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../adversarial/context/mandarin_TOXICITY.csv', header=None)
df.columns = ['lyrics', 'label', 'score']

df['label'] = df['label'].astype(str).str.strip().str.upper()
df['score'] = pd.to_numeric(df['score'], errors='coerce')

def classify(label, score):
    if label == 'T' and score >= 0.50:
        return 'True Positive'
    elif label == 'T' and score < 0.50:
        return 'False Positive'
    elif label == 'F' and score < 0.50:
        return 'True Negative'
    elif label == 'F' and score >= 0.50:
        return 'False Negative'
    else:
        return None  

df['classification'] = df.apply(lambda row: classify(row['label'], row['score']), axis=1)
df = df[df['classification'].notnull()]  # remove unknowns

language_map = {0: 'Chinese', 1: 'Mixed', 2: 'English'}
df['language_type'] = df.index % 3
df['language_type'] = df['language_type'].map(language_map)

fig, axes = plt.subplots(2, 3, figsize=(15, 6))
fig.suptitle('Toxicity Score by Language Type and Classification', fontsize=16)

palette = {
    'True Positive': 'green',
    'True Negative': 'red',
    'False Positive': 'blue',
    'False Negative': 'purple'
}
colors = {'Chinese': 'green', 'Mixed': 'orange', 'English': 'blue'}
bins = np.linspace(0, 1, 11)
languages = ['Chinese', 'Mixed', 'English']

for i, lang in enumerate(languages):
    ax = axes[0, i]
    subset = df[df['language_type'] == lang]
    sns.stripplot(
        x='classification',
        y='score',
        data=subset,
        palette=palette,
        jitter=False,
        alpha=0.7,
        ax=ax,
        order=['True Negative', 'False Positive', 'True Positive', 'False Negative']
    )
    ax.set_title(f'{lang}')
    ax.set_xlabel('Classification')
    ax.set_ylabel('Toxicity Score')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)

for i, lang in enumerate(languages):
    ax = axes[1, i]
    subset = df[df['language_type'] == lang]
    ax.hist(
        subset['score'],
        bins=bins,
        color=colors[lang],
        edgecolor='black',
        alpha=0.8
    )
    ax.set_title(f'{lang} – Score Distribution')
    ax.set_xlabel('Toxicity Score')
    ax.set_ylabel('Count')
    ax.set_ylim(0, 8)  # Uniform y-axis across all histograms
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('toxicity_3x2_grid.png', dpi=300, bbox_inches='tight')
plt.show()
