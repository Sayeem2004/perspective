import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# aa_dataset = '../classification/race/unlabeled-AA-1100_TOXICITY.csv'
# hispanic_dataset = '../classification/race/unlabeled-HISPANIC-1100_TOXICITY.csv'
# white_dataset = '../classification/race/unlabeled-WHITE-1100_TOXICITY.csv'
# other_dataset = '../classification/race/unlabeled-OTHER-1100_TOXICITY.csv'

# genders = ['FEMALE', 'MALE']
topics_labeled = ['labeled-Adam Levine-50','labeled-German Shepherd-50','labeled-My Little Pony-50','labeled-Border Collie-50','labeled-Guitar-50','labeled-Stephen King-50','labeled-Bruno Mars-50','labeled-Iguana-50','labeled-Donna Karan-50','labeled-Justin Bieber-50']
# 'labeled-Adam Levine-50.csv'    'labeled-German Shepherd-50.csv'  'labeled-My Little Pony-50.csv'    unlabeled-Bruno_Mars-415.csv        unlabeled-Iguana-281.csv
# 'labeled-Border Collie-50.csv'   labeled-Guitar-50.csv            'labeled-Stephen King-50.csv'      unlabeled-Donna_Karan-229.csv       unlabeled-Justin_Bieber-340.csv
# 'labeled-Bruno Mars-50.csv'      labeled-Iguana-50.csv             unlabeled-Adam_Levine-214.csv     unlabeled-German_Shepherd-306.csv   unlabeled-My_Little_Pony-191.csv
# 'labeled-Donna Karan-50.csv'    'labeled-Justin Bieber-50.csv'     unlabeled-Border_Collie-227.csv   unlabeled-Guitar-681.csv            unlabeled-Stephen_King-234.csv
topics_unlabeled = ['unlabeled-Adam_Levine-214','unlabeled-German_Shepherd-306','unlabeled-My_Little_Pony-191','unlabeled-Border_Collie-227','unlabeled-Guitar-681','unlabeled-Stephen_King-234','unlabeled-Bruno_Mars-415','unlabeled-Iguana-281','unlabeled-Donna_Karan-229','unlabeled-Justin_Bieber-340']

communities_humans = ['gaming.csv','music.csv','novel_labeled.csv','tech_labeled.csv']
communities_perspective = ['gaming_TOXICITY.csv','music_TOXICITY.csv','novel_TOXICITY.csv','tech_TOXICITY.csv']

def get_human_probabilities():
    """
    Get the human probabilities from the dataset.
    """
    probs = [0]*14
    for i, topic in enumerate(topics_labeled):
        count=0
        with open(f'../dataset/topic/{topic}.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # print(line)
                if line.endswith('T\n'):
                    count += 1
        probs[i] = count / 50.0

    for i, topic in enumerate(communities_humans):
        count=0
        with open(f'../adversarial/community/{topic}', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # print(line)
                if line.endswith('T\n'):
                    count += 1
        probs[i+10] = count / 100.0
    return probs

def get_perspective_probabilities():
    """
    Get the perspective probabilities from the dataset.
    """
    perspective_probs = [0]*14
    for i, topic in enumerate(topics_labeled):
        count=0
        with open(f'../classification/topic/{topic}_TOXICITY.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # find the last comma
                last_comma = line.rfind(',')
                # get the last part of the line
                last_part = line[last_comma+1:]
                # print(last_part)
                if float(last_part) >= 0.5:
                    count += 1
        perspective_probs[i] = count / float(50)

    for i, topic in enumerate(communities_perspective):
        count=0
        with open(f'../classification/community/{topic}', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # find the last comma
                last_comma = line.rfind(',')
                # get the last part of the line
                last_part = line[last_comma+1:]
                if float(last_part) >= 0.5:
                    count += 1
        perspective_probs[i+10] = count / 100.0
    return perspective_probs

if __name__ == '__main__':
    human_probs = get_human_probabilities()
    human_probs[1] = 1
    perspective_probs = get_perspective_probabilities()

    human_probs = human_probs[10:]
    perspective_probs = perspective_probs[10:]

    # print('Human probabilities: ', get_human_probabilities())
    # print('Perspective probabilities 100: ', get_perspective_probabilities())
    # print('Bias: ', np.array(get_perspective_probabilities()) / np.array(get_human_probabilities()))

    # create a bar graph of bias
    # bias_1000 = np.array(get_perspective_probabilities(1000)) / np.array(get_human_probabilities())
    # bias_100 = np.array(get_perspective_probabilities(100)) / np.array(get_human_probabilities())
    # bias_100[3] += 0.05

    bias = np.array(perspective_probs) / np.array(human_probs)

    sub_categories = ['100']

    # Data for each sub-category (same order as sub_categories)
    values = [
        bias,
    ]

    values = np.array(values)
    values = values.T

    # Plot settings
    bar_width = 0.25
    x = np.arange(4)
    colors = ['#1f77b4', '#f4a261', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group of bars
    for i in range(len(sub_categories)):
        ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=sub_categories[i], color=colors[i])
        # Add value labels on top of bars
        for j in range(4):
            ax.text(x[j] + i * bar_width, values[j, i] + 0.01, f'{values[j, i]:.2f}',
                    ha='center', va='bottom', fontsize=16)

    ax.axhline(y=1, color='k', linestyle='--', linewidth=2)
    ax.set_title('Community Bias', fontsize=24)
    ax.set_ylabel('Toxicity Ratio', fontsize=24)
    # ax.set_xlabel('Race Type', fontsize=24)
    ax.set_xticks(x)
    # ax.set_xticklabels(['African\nAmerican', 'Hispanic', 'White', 'Other'], fontsize=24)
    ax.set_xticklabels(['GAMING', 'MUSIC', 'NOVEL', 'TECH'], fontsize=24)
    # ax.set_xticklabels(genders, fontsize=24)
    ax.legend(fontsize=24)

    # Grid and layout
    ax.grid(axis='y', linestyle='-', linewidth=0.8, alpha=0.9)
    plt.tight_layout()

    plt.savefig('./topic_graph.png', dpi=300)
    # plt.show()

    plt.close()
