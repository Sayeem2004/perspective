from googleapiclient import discovery
from dotenv import load_dotenv
import argparse
import time
import os

if __name__ == "__main__":
    # Arguments Parsing
    parser = argparse.ArgumentParser(description="Compare manual vs API classification.")
    parser.add_argument("--file", type=str, default="comments.txt", help="Path to the CSV file containing API labels.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification.")
    args = parser.parse_args()

    # Tracked Statistics
    api_true, api_false = 0, 0

    # Iterating Through CSV files
    with open(args.file, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Splitting Lines
            print(f"Analyzing: {i + 1}/{len(lines)}")
            split = line.split(",")

            # Adding To Statistics
            api_true += (float(split[-1]) >= args.threshold)
            api_false += (float(split[-1]) < args.threshold)
            time.sleep(0.01)

    # Printing Results
    print(f'API Toxicity: {api_true}/{api_true+api_false}')
