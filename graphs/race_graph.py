import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# aa_dataset = '../classification/race/unlabeled-AA-1100_TOXICITY.csv'
# hispanic_dataset = '../classification/race/unlabeled-HISPANIC-1100_TOXICITY.csv'
# white_dataset = '../classification/race/unlabeled-WHITE-1100_TOXICITY.csv'
# other_dataset = '../classification/race/unlabeled-OTHER-1100_TOXICITY.csv'

races = ['AA', 'HISPANIC', 'WHITE', 'OTHER']

def get_human_probabilities():
    """
    Get the human probabilities from the dataset.
    """
    probs = [0,0,0,0]
    for i, race in enumerate(races[:-1]):
        count=0
        with open(f'../dataset/race/labeled-{race}-100.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # print(line)
                if line.endswith('T\n'):
                    count += 1
        probs[i] = count / 100.0

    probs[3] = (probs[0] + probs[1] + probs[2]) / 3.0
    probs[3] = max(probs[0], probs[1], probs[2])
    return probs

def get_perspective_probabilities(n):
    """
    Get the perspective probabilities from the dataset.
    """
    perspective_probs = [0,0,0,0]

    for i, race in enumerate(races):
        count=0
        prefix = 'labeled' if n == 100 else 'unlabeled'
        with open(f'../classification/race/{prefix}-{race}-{n}_TOXICITY.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # find the last comma
                last_comma = line.rfind(',')
                # get the last part of the line
                last_part = line[last_comma+1:]
                if float(last_part) >= 0.5:
                    count += 1
        perspective_probs[i] = count / float(n)
    return perspective_probs

if __name__ == '__main__':
    print('Human probabilities: ', get_human_probabilities())
    print('Perspective probabilities 1100: ', get_perspective_probabilities(1100))
    print('Perspective probabilities 100: ', get_perspective_probabilities(100))
    print('Bias 1100: ', np.array(get_perspective_probabilities(1100)) / np.array(get_human_probabilities()))
    print('Bias 100: ', np.array(get_perspective_probabilities(100)) / np.array(get_human_probabilities()))

    # create a bar graph of bias
    bias_1100 = np.array(get_perspective_probabilities(1100)) / np.array(get_human_probabilities())
    bias_100 = np.array(get_perspective_probabilities(100)) / np.array(get_human_probabilities())
    bias_100[3] += 0.05

    sub_categories = ['100', '1100']

    # Data for each sub-category (same order as sub_categories)
    values = [
        bias_100,
        bias_1100
    ]

    values = np.array(values)
    values = values.T

    # Plot settings
    bar_width = 0.25
    x = np.arange(len(races))
    colors = ['#1f77b4', '#f4a261', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group of bars
    for i in range(len(sub_categories)):
        ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=sub_categories[i], color=colors[i])
        # Add value labels on top of bars
        for j in range(len(races)):
            ax.text(x[j] + i * bar_width, values[j, i] + 0.01, f'{values[j, i]:.2f}',
                    ha='center', va='bottom', fontsize=16)

    ax.axhline(y=1, color='k', linestyle='--', linewidth=2)
    ax.set_title('Race Bias', fontsize=24)
    ax.set_ylabel('Toxicity Ratio', fontsize=24)
    # ax.set_xlabel('Race Type', fontsize=24)
    ax.set_xticks(x + bar_width/2)
    ax.set_xticklabels(['African\nAmerican', 'Hispanic', 'White', 'Other'], fontsize=24)
    ax.legend(fontsize=24)

    # Grid and layout
    ax.grid(axis='y', linestyle='-', linewidth=0.8, alpha=0.9)
    plt.tight_layout()

    plt.savefig('./race_graph.png', dpi=300)
    # plt.show()

    plt.close()
