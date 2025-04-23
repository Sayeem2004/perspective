import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../../perspective/classification/context/mandarin_TOXICITY.csv', header=None)
df.columns = ['sentence', 'label', 'score']

df['classification'] = 'unknown'
df.loc[(df['label'] == 'T') & (df['score'] >= 0.50), 'classification'] = 'True Positive'
df.loc[(df['label'] == 'T') & (df['score'] < 0.50), 'classification'] = 'False Positive'
df.loc[(df['label'] == 'F') & (df['score'] < 0.50), 'classification'] = 'True Negative'
df.loc[(df['label'] == 'F') & (df['score'] >= 0.50), 'classification'] = 'False Negative'

language_map = {0: 'Chinese', 1: 'Mixed', 2: 'English'}
df['language_type'] = df.index % 3
df['language_type'] = df['language_type'].map(language_map)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Toxicity Score Distribution by Language Type', fontsize=16)

languages = ['Chinese', 'Mixed', 'English']
colors = {'Chinese': 'green', 'Mixed': 'orange', 'English': 'blue'}
bins = np.linspace(0, 1, 11)

max_count = 0
for lang in languages:
    counts, _ = np.histogram(df[df['language_type'] == lang]['score'], bins=bins)
    max_count = max(max_count, counts.max())

for i, lang in enumerate(languages):
    ax = axes[i]
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
    ax.set_ylim(0, max_count + 1)  # ensure consistent y-axis
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('toxicity_histograms_only.png', dpi=300, bbox_inches='tight')
plt.show()
