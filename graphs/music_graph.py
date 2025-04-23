import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('../adversarial/community/gaming_TOXICITY.csv', header=None)

# Assign column names
df.columns = ['lyrics', 'label', 'score']

# Sanitize label and score values
df['label'] = df['label'].astype(str).str.strip().str.upper()   # Clean label
df['score'] = pd.to_numeric(df['score'], errors='coerce')       # Ensure score is float

print("Starting scatterplot")

# Create classification categories
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
        return 'unknown'

df['classification'] = df.apply(lambda row: classify(row['label'], row['score']), axis=1)

# Debugging block to find 'unknown' results
unknown_rows = df[df['classification'] == 'unknown']
print("Rows with 'unknown' classification:")
print(unknown_rows)
print(f"Total 'unknown' rows: {len(unknown_rows)}")
print("Unique values in classification column:")
print(df['classification'].unique())

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Toxicity Score Analysis', fontsize=16)

# Define color palette
palette = {
    'True Positive': 'green',
    'True Negative': 'red',
    'False Positive': 'blue',
    'False Negative': 'purple',
    'unknown': 'gray'
}

# 1. Strip plot
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