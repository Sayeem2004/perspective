from googleapiclient import discovery
from dotenv import load_dotenv
import argparse
import time
import os

if __name__ == "__main__":
    # Arguments Parsing
    parser = argparse.ArgumentParser(description="Compare manual vs API classification.")
    parser.add_argument("--file_man", type=str, default="comments.txt", help="Path to the CSV file containing manual labels.")
    parser.add_argument("--file_api", type=str, default="comments.txt", help="Path to the CSV file containing API labels.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification.")
    args = parser.parse_args()

    # Tracked Statistics
    man_true, man_false = 0, 0
    api_true, api_false = 0, 0

    # Iterating Through CSV files
    with open(args.file_man, "r") as file_man:
        with open(args.file_api, "r") as file_api:
            lines = zip(file_man.readlines(), file_api.readlines())
            for i, (line_man, line_api) in enumerate(lines):
                # Splitting Lines
                print(f"Comparing: {i + 1}")
                split_man = line_man.split("\t")
                split_api = line_api.split(",")

                # Adding To Statistics
                man_true += ('T' in split_man[-1])
                man_false += ('F' in split_man[-1])
                api_true += (float(split_api[-1]) >= args.threshold)
                api_false += (float(split_api[-1]) < args.threshold)
                time.sleep(0.01)

                # Printing Differences
                if 'T' in split_man[-1] and float(split_api[-1]) < args.threshold:
                    print(f'Manual True, API False, {split_man[0]}')
                if 'F' in split_man[-1] and float(split_api[-1]) >= args.threshold:
                    print(f'Manual False, API True, {split_man[0]}')

    # Printing Results
    print(f'Manual Toxicity: {man_true}/{man_true+man_false}')
    print(f'API Toxicity: {api_true}/{api_true+api_false}')
