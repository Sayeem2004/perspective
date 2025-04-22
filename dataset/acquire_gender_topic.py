from datasets import load_dataset
import numpy as np
import os

def collect_data(database, output_dir, find_male):
    threshold = 0.95
    def criterion(score, find_male): return score > threshold if find_male else (1 - score) > threshold

    data = []
    os.makedirs(output_dir, exist_ok=True)
    for xData in dataset:
        text, score = xData['text'], xData['gender']
        if score == find_male:
            data.append((text, score))
        
        if len(data) >= 10000: break
        
    # gender = 'MALE' if find_male else 'FEMALE'
    if find_male == 0: gender = 'NEUTRAL'
    elif find_male == 1: gender = 'FEMALE'
    elif find_male == 2: gender = 'MALE'
    
    with open(os.path.join(output_dir, f"unlabeled-{gender}-{len(data)}.csv"), 'w') as f:
        for text, score in data:
            f.write(f"{text}\t{score}\n")
    print(f"Found {len(data)} data points")
    
    # subsample 100 for human labeling
    sample_size = 100
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    sample_data = [data[i] for i in indices]
    with open(os.path.join(output_dir, f"labeled-{gender}-{sample_size}.csv"), 'w') as f:
        for text, score in sample_data:
            f.write(f"{text}\t{score}\n")
    print(f"Subsampled {sample_size} data points for human labeling")

def combine_dataset(dataset):
    combined_data = []
    for data in dataset['train']: combined_data.append({'text': data['text'], 'gender': data['gender']})
    for data in dataset['test']: combined_data.append({'text': data['text'], 'gender': data['gender']})
    for data in dataset['validation']: combined_data.append({'text': data['text'], 'gender': data['gender']})
    return combined_data

if __name__ == "__main__":
    dataset = load_dataset("facebook/md_gender_bias", "funpedia")
    # dataset = dataset['train']
    # concatenate all splits
    dataset = combine_dataset(dataset)
    collect_data(dataset, 'gender_topic', find_male=0)
    collect_data(dataset, 'gender_topic', find_male=1)
    collect_data(dataset, 'gender_topic', find_male=2)