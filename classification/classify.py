from googleapiclient import discovery
from dotenv import load_dotenv
import argparse
import time
import os

if __name__ == "__main__":
    # Arguments Parsing
    parser = argparse.ArgumentParser(description="Classify comments using Google API.")
    parser.add_argument("--file", type=str, default="comments.txt", help="Path to the CSV file containing comments.")
    parser.add_argument("--category", type=str, default="TOXICITY", help="Class to classify the comments.")
    args = parser.parse_args()

    # Creating API Client
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    # Creating output file
    filename = args.file.split(".")[0]
    output_file = f"{filename}_{args.category}.csv"

    # Iterating through CSV file
    with open(args.file, "r") as file:
        with open(output_file, "w") as output:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Creating Request
                print(f"Analyzing: {i + 1}/{len(lines)}")
                comment = line.split("\t")[0]
                analyze_request = {
                    'comment': {'text': comment.strip()},
                    'requestedAttributes': {args.category: {}},
                    'languages': ['en', 'es']
                }

                # Sending Request + Results
                response = client.comments().analyze(body=analyze_request).execute()
                score = response["attributeScores"][args.category]["summaryScore"]["value"]
                output.write(f"{comment.strip()},{score}\n")
                output.flush()
                time.sleep(1)
