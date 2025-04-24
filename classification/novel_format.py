import pandas as pd
import csv

output_file = 'novel_TOXICITY_fixed.csv'

try:
    processed_rows = []
    
    with open('classification/community/novel_TOXICITY.csv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            last_comma_pos = line.rfind(',')
            if last_comma_pos == -1:
                processed_rows.append([line])
                continue
                
            second_last_comma_pos = line.rfind(',', 0, last_comma_pos)
            if second_last_comma_pos == -1:
                text = line[:last_comma_pos]
                label_score = line[last_comma_pos+1:]
                processed_rows.append([text, label_score])
                continue
            
            text = line[:second_last_comma_pos]
            label = line[second_last_comma_pos+1:last_comma_pos]
            score = line[last_comma_pos+1:]
            
            processed_rows.append([text, label, score])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)
    
    print(f"Processing complete. Output saved to {output_file}")
    print(f"First few rows of processed data:")
    
    for i, row in enumerate(processed_rows[:5]):
        print(f"Row {i+1}: {row}")
    
except Exception as e:
    print(f"Error processing file: {e}")