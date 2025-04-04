from googleapiclient import discovery
from dotenv import load_dotenv
import json
import os

if __name__ == "__main__":
    # Getting API Key
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    # Creating API Client
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    # Creating Request
    analyze_request = {
        'comment': { 'text': 'I fucking love you :)' },
        'requestedAttributes': {'TOXICITY': {}}
    }

    # Sending Request + Results
    response = client.comments().analyze(body=analyze_request).execute()
    print(json.dumps(response, indent=2))
