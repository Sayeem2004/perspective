import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

print("Starting data loading...")
df = pd.read_csv('classification/community/novel_TOXICITY.csv', header=None)
print(f"CSV loaded in {time.time() - start_time:.2f} seconds")
print(f"Data shape: {df.shape}")

df.columns = ['lyrics', 'label', 'score']

print("Cleaning data...")
df['label'] = df['label'].astype(str).str.strip().str.upper()
df['score'] = pd.to_numeric(df['score'], errors='coerce')

df['text_length'] = df['lyrics'].str.len()

print("Creating classifications...")
def classify(label, score):
    if label == 'T' and score >= 0.50:
        return 'True Positive'
    elif label == 'F' and score >= 0.50:
        return 'False Positive'
    elif label == 'T' and score < 0.50:
        return 'False Negative'
    elif label == 'F' and score < 0.50:
        return 'True Negative'
    else:
        return 'Unknown'

df['classification'] = df.apply(lambda row: classify(row['label'], row['score']), axis=1)

colors = {
    'True Positive': 'green',
    'False Positive': 'red',
    'False Negative': 'orange',
    'True Negative': 'blue',
    'Unknown': 'gray'
}

print("Creating plot...")
fig, ax = plt.subplots(figsize=(14, 10))

plt.axhline(y=0.5, color='black', linestyle='-', linewidth=2)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)

toxic_lengths = df[df['label'] == 'T']['text_length']
nontoxic_lengths = df[df['label'] == 'F']['text_length']

toxic_quantiles = [np.percentile(toxic_lengths, q) for q in [0, 25, 50, 75, 100]]
nontoxic_quantiles = [np.percentile(nontoxic_lengths, q) for q in [0, 25, 50, 75, 100]]

toxic_quantiles = [int(q) for q in toxic_quantiles]
nontoxic_quantiles = [int(q) for q in nontoxic_quantiles]

# Create normalized positions based on text length percentiles within each class
# For toxic content (T), use position from -1 to 0 (starting from largest to smallest)
# For non-toxic content (F), use position from 0 to 1 (starting from smallest to largest)

def map_to_position(row):
    if row['label'] == 'T':
        min_val = min(toxic_lengths)
        max_val = max(toxic_lengths)
        return (row['text_length'] - min_val) / (max_val - min_val)
    else:
        min_val = min(nontoxic_lengths)
        max_val = max(nontoxic_lengths)
        return -1 * (row['text_length'] - min_val) / (max_val - min_val)

df['position'] = df.apply(map_to_position, axis=1)

for cat, color in colors.items():
    cat_data = df[df['classification'] == cat]
    if not cat_data.empty:
        plt.scatter(
            cat_data['position'], 
            cat_data['score'],
            color=color,
            alpha=0.7,
            s=80,
            label=cat
        )

plt.ylim(-0.05, 1.05)
plt.xlim(-1.05, 1.05)

plt.xlabel("TEXT LENGTH (CHARACTERS)", fontsize=14, fontweight='bold')
plt.ylabel("MODEL TOXICITY SCORE", fontsize=14, fontweight='bold')

plt.text(-0.5, 1.02, "C - FALSE POSITIVES", fontsize=14, fontweight='bold', ha='center')
plt.text(0.5, 1.02, "A - TRUE POSITIVES", fontsize=14, fontweight='bold', ha='center')
plt.text(-0.5, -0.02, "D - TRUE NEGATIVES", fontsize=14, fontweight='bold', ha='center')
plt.text(0.5, -0.02, "B - FALSE NEGATIVES", fontsize=14, fontweight='bold', ha='center')
plt.title("NOVEL CONTENT TOXICITY DETECTION PERFORMANCE", fontsize=18, fontweight='bold')

x_ticks = [-1, -0.5, 0, 0.5, 1]
x_labels = [
    f"{nontoxic_quantiles[4]} chars (Non-Toxic)", 
    f"{nontoxic_quantiles[2]} chars (Non-Toxic)", 
    f"{int((nontoxic_quantiles[0] + toxic_quantiles[0])/2)} chars", 
    f"{toxic_quantiles[2]} chars (Toxic)", 
    f"{toxic_quantiles[4]} chars (Toxic)"
]
plt.xticks(x_ticks, x_labels, fontsize=10)
y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
plt.yticks(y_ticks, ['0.0', '0.25', '0.5', '0.75', '1.0'])

plt.legend(title="Classification", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.grid(True, linestyle='--', alpha=0.3)

print("Finalizing plot...")
plt.tight_layout()
print(f"Attempting to save plot...")
plt.savefig('graphs/novel_graph.png', dpi=300, bbox_inches='tight')
print(f"Plot saved successfully")

try:
    plt.show()
except Exception as e:
    print(f"Could not display plot interactively: {e}")

print(f"Total execution time: {time.time() - start_time:.2f} seconds")