import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CSV file
df = pd.read_csv('classification/community/novel_TOXICITY.csv', header=None)

# Assign column names
df.columns = ['lyrics', 'label', 'score']

# Sanitize label and score values
df['label'] = df['label'].astype(str).str.strip().str.upper()   # Clean label
df['score'] = pd.to_numeric(df['score'], errors='coerce')       # Ensure score is float

# Create classification categories
def classify(label, score):
    if label == 'T' and score >= 0.50:
        return 'True Positive'
    elif label == 'T' and score < 0.50:
        return 'False Negative'
    elif label == 'F' and score < 0.50:
        return 'True Negative'
    elif label == 'F' and score >= 0.50:
        return 'False Positive'
    else:
        return 'Unknown'

df['classification'] = df.apply(lambda row: classify(row['label'], row['score']), axis=1)

# Count the occurrences of each classification
classification_counts = df['classification'].value_counts().to_dict()

# Create a 2x2 grid for the Punnett square
fig, ax = plt.subplots(figsize=(10, 8))

# Define the Punnett square
punnett_data = np.zeros((2, 2))
punnett_labels = [['True Positive', 'False Positive'], 
                 ['False Negative', 'True Negative']]

# Fill with actual counts
for i in range(2):
    for j in range(2):
        label = punnett_labels[i][j]
        count = classification_counts.get(label, 0)
        punnett_data[i, j] = count

# Create a custom colormap with better contrast
# Using a custom colormap for better visibility
custom_cmap = plt.cm.get_cmap('Oranges', 256)  # Change to a more visible colormap

# Create heatmap (Punnett square)
im = ax.imshow(punnett_data, cmap=custom_cmap)

# Add labels to the cells
for i in range(2):
    for j in range(2):
        label = punnett_labels[i][j]
        count = punnett_data[i, j]
        percentage = (count / len(df)) * 100
        text = f"{label}\n{int(count)} ({percentage:.1f}%)"
        ax.text(j, i, text, ha='center', va='center', 
                fontsize=12, color='black', fontweight='bold')

# Add axis labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Novel Toxicity Classification Confusion Matrix', fontsize=16)

# Add legend for classifications
x_label = "Model Score < 0.50 | Model Score ≥ 0.50"
y_label = "Actual: Not Toxic (F) | Actual: Toxic (T)"
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12, rotation=90, labelpad=15)

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Number of Instances', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('graphs/novel_matrix.png', dpi=300, bbox_inches='tight')
plt.show()