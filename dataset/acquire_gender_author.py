from datasets import load_dataset
import numpy as np
import os

def collect_data(database, output_dir, find_male):
    threshold = 0.95
    def criterion(score, find_male): return score > threshold if find_male else (1 - score) > threshold

    data = []
    os.makedirs(output_dir, exist_ok=True)
    for xData in dataset:
        text, score = xData['text'], xData['binary_score']
        # if criterion(score, find_male):
        if xData['binary_label'] == find_male:
            data.append((text, score))
        
        if len(data) >= 10000: break
        
    gender = 'MALE' if find_male else 'FEMALE'
    with open(os.path.join(output_dir, f"unlabeled-{gender}-{10000}.csv"), 'w') as f:
        for text, score in data:
            f.write(f"{text}\n")
    print(f"Found {len(data)} data points")
    
    # subsample 100 for human labeling
    sample_size = 100
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    sample_data = [data[i] for i in indices]
    with open(os.path.join(output_dir, f"labeled-{gender}-{sample_size}.csv"), 'w') as f:
        for text, score in sample_data:
            f.write(f"{text}\n")
    print(f"Subsampled {sample_size} data points for human labeling")
    
if __name__ == "__main__":
    dataset = load_dataset("facebook/md_gender_bias", "yelp_inferred")
    dataset = dataset['train']
    collect_data(dataset, 'gender_author', find_male=True)
    collect_data(dataset, 'gender_author', find_male=False)