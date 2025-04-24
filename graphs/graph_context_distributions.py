import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD
df = pd.read_csv('../classification/context/mandarin_TOXICITY.csv', header=None)
=======
# Load and process data
df = pd.read_csv('../../perspective/classification/context/mandarin_TOXICITY.csv', header=None)
>>>>>>> ee40774f4b182448705d000cd9aaef38973a416b
df.columns = ['sentence', 'label', 'score']

df['classification'] = 'unknown'
df.loc[(df['label'] == 'T') & (df['score'] >= 0.50), 'classification'] = 'True Positive'
df.loc[(df['label'] == 'T') & (df['score'] < 0.50), 'classification'] = 'False Positive'
df.loc[(df['label'] == 'F') & (df['score'] < 0.50), 'classification'] = 'True Negative'
df.loc[(df['label'] == 'F') & (df['score'] >= 0.50), 'classification'] = 'False Negative'

language_map = {0: 'Chinese', 1: 'Mixed', 2: 'English'}
df['language_type'] = df.index % 3
df['language_type'] = df['language_type'].map(language_map)

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Toxicity Score Distribution by Language Type', fontsize=20)

languages = ['Chinese', 'Mixed', 'English']
colors = {'Chinese': 'green', 'Mixed': 'orange', 'English': 'blue'}
bins = np.linspace(0, 1, 11)

# Compute max height for y-axis
max_count = 0
for lang in languages:
    counts, _ = np.histogram(df[df['language_type'] == lang]['score'], bins=bins)
    max_count = max(max_count, counts.max())

# Plot histograms
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
<<<<<<< HEAD
    ax.set_title(f'{lang} – Score Distribution', fontsize=16)
    ax.set_xlabel('Toxicity Score')
    ax.set_ylabel('Count')
    ax.set_ylim(0, max_count + 1)  # ensure consistent y-axis
=======
    ax.set_title(f'{lang}', fontsize=20)
    ax.set_xlabel('Toxicity Score', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_ylim(0, max_count + 1)
    ax.tick_params(axis='both', labelsize=12)
>>>>>>> ee40774f4b182448705d000cd9aaef38973a416b
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('toxicity_histograms_only_larger_fonts.png', dpi=300, bbox_inches='tight')
plt.show()
