import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of file paths
file_paths = [
    'classification/community/music_TOXICITY.csv',
    'classification/community/gaming_TOXICITY.csv',
    'classification/community/novel_TOXICITY.csv',
    'classification/community/tech_TOXICITY.csv'  # Adjust if filename is different
]

# List of titles for each subplot
titles = ['Music Content', 'Gaming Content', 'Novel Content', 'Tech Content']  # Adjust as needed

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten the 2x2 array to make indexing easier
fig.suptitle('Toxicity Score Distribution by Content Type', fontsize=18)

# Colors for each content type
colors = {'Music Content': 'green', 'Gaming Content': 'red', 'Novel Content': 'blue', 'Tech Content': 'orange'}

# Define bins for histograms
bins = np.linspace(0, 1, 11)

# Find max count across all datasets for consistent y-axis scaling
max_count = 0
dfs = []

# First pass: load data and find max count
for i, file_path in enumerate(file_paths):
    try:
        # Load file
        df = pd.read_csv(file_path, header=None)
        df.columns = ['sentence', 'label', 'score']
        dfs.append(df)
        
        # Calculate histogram data
        counts, _ = np.histogram(df['score'], bins=bins)
        max_count = max(max_count, counts.max())
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Create empty dataframe if file can't be loaded
        dfs.append(pd.DataFrame({'sentence': [], 'label': [], 'score': []}))

# Second pass: create histograms
for i, (df, title) in enumerate(zip(dfs, titles)):
    ax = axes[i]
    
    # Create histogram
    ax.hist(
        df['score'],
        bins=bins,
        color=colors[title],
        edgecolor='black',
        alpha=0.8
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Toxicity Score')
    ax.set_ylabel('Count')
    
    # Set y-axis limit based on data
    if not df.empty:
        counts, _ = np.histogram(df['score'], bins=bins)
        y_max = max(counts.max() + 2, 8)  # Ensure minimum height of 8 for visual consistency
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, 8)  # Default if no data
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove box at top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('graphs/community_histograms.png', dpi=300, bbox_inches='tight')
plt.show()