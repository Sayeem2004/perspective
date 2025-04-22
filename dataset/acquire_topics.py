from datasets import load_dataset
import numpy as np
import os

def collect_data(database, output_dir, topic):
    threshold = 0.95

    data = []
    os.makedirs(output_dir, exist_ok=True)
    for xData in dataset:
        text, score = xData['text'], xData['chosen_topic']
        if score == topic:
            data.append((text, score))
        
        if len(data) >= 10000: break
        
    if topic == 'My Little Pony: Friendship Is Magic fandom': title_topic = 'My Little Pony'
    else: title_topic = topic
    
    with open(os.path.join(output_dir, f"unlabeled-{title_topic.replace(' ', '_')}-{len(data)}.csv"), 'w') as f:
        for text, score in data:
            f.write(f"{text}\n")
    print(f"Found {len(data)} data points for {topic}")
    
    # subsample 100 for human labeling
    sample_size = 50
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    sample_data = [data[i] for i in indices]
    with open(os.path.join(output_dir, f"labeled-{title_topic}-{sample_size}.csv"), 'w') as f:
        for text, score in sample_data:
            f.write(f"{text}\n")
    print(f"Subsampled {sample_size} data points for human labeling")

def combine_dataset(dataset):
    combined_data = []
    for data in dataset['train']: combined_data.append({'text': data['text'], 'chosen_topic': data['chosen_topic']})
    for data in dataset['test']: combined_data.append({'text': data['text'], 'chosen_topic': data['chosen_topic']})
    for data in dataset['validation']: combined_data.append({'text': data['text'], 'chosen_topic': data['chosen_topic']})
    return combined_data

if __name__ == "__main__":
    topics = ['Guitar', 'Bruno Mars', 'Justin Bieber', 'German Shepherd', 'Iguana', 'Donna Karan', 'Border Collie', 'Stephen King', 'Adam Levine', 'My Little Pony: Friendship Is Magic fandom']
    dataset = load_dataset("facebook/md_gender_bias", "wizard")
    dataset = combine_dataset(dataset)
    for topic in topics:
        collect_data(dataset, 'topic', topic)