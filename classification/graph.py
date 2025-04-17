import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('classification/niche/music_TOXICITY.csv', header=None)

# Assign column names based on the structure you described
df.columns = ['lyrics', 'label', 'score']

print("Starting scatterplot")
# Create classification categories
df['classification'] = 'unknown'
df.loc[(df['label'] == 'T') & (df['score'] >= 0.50), 'classification'] = 'True Positive'
df.loc[(df['label'] == 'T') & (df['score'] < 0.50), 'classification'] = 'False Positive'
df.loc[(df['label'] == 'F') & (df['score'] < 0.50), 'classification'] = 'True Negative'
df.loc[(df['label'] == 'F') & (df['score'] >= 0.50), 'classification'] = 'False Negative'

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Toxicity Score Analysis', fontsize=16)

# 1. Strip plot (jittered points by category)
# Define color palette
palette = {
    'True Positive': 'green',
    'True Negative': 'red',
    'False Positive': 'blue',
    'False Negative': 'purple'
}

# Create a strip plot
sns.stripplot(
    x='classification',
    y='score',
    data=df,
    palette=palette,
    jitter=True,
    alpha=0.7,
    ax=ax1
)

ax1.set_xlabel('Classification')
ax1.set_ylabel('Toxicity Score')
ax1.set_title('Toxicity Scores by Classification Category')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 1)

print("Starting histogram")
# 2. Histogram Plot
bins = np.linspace(0, 1, 11)
ax2.hist(df['score'], bins=bins, edgecolor='black')
ax2.set_xlabel('Toxicity Score Range')
ax2.set_ylabel('Number of Data Points')
ax2.set_title('Distribution of Toxicity Scores')
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('toxicity_visualization.png', dpi=300, bbox_inches='tight')
plt.show()