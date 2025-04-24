import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of file paths
file_paths = [
    'classification/community/music_TOXICITY.csv',
    'classification/community/gaming_TOXICITY.csv',
    'classification/community/novel_TOXICITY.csv',
    'classification/community/tech_TOXICITY.csv'
]

# List of titles for each subplot
titles = ['Music', 'Gaming', 'Novel', 'Tech']

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Increased figure size for readability
axes = axes.flatten()
# fig.suptitle('Toxicity Score Distribution by Content Type', fontsize=30)

# Colors for each content type
colors = {'Music': 'green', 'Gaming': 'red', 'Novel': 'blue', 'Tech': 'orange'}

# Define bins for histograms
bins = np.linspace(0, 1, 11)

# Find max count across all datasets for consistent y-axis scaling
max_count = 0
dfs = []

# First pass: load data and find max count
for i, file_path in enumerate(file_paths):
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['sentence', 'label', 'score']
        dfs.append(df)

        counts, _ = np.histogram(df['score'], bins=bins)
        max_count = max(max_count, counts.max())
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        dfs.append(pd.DataFrame({'sentence': [], 'label': [], 'score': []}))

# Second pass: create histograms
for i, (df, title) in enumerate(zip(dfs, titles)):
    ax = axes[i]

    ax.hist(
        df['score'],
        bins=bins,
        color=colors[title],
        edgecolor='black',
        alpha=0.8
    )

    ax.set_title(title, fontsize=50)
    ax.set_xlabel('Toxicity Score', fontsize=24)
    ax.set_ylabel('Count', fontsize=24)
    ax.tick_params(axis='both', labelsize=30)

    if not df.empty:
        counts, _ = np.histogram(df['score'], bins=bins)
        y_max = max(counts.max() + 2, 8)
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, 8)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('graphs/community_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
