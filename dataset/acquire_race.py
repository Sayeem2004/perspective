import pandas as pd
import numpy as np
import multiprocessing

database = "/mnt/c/Users/raigo/Downloads/TwitterAAE-full-v1/TwitterAAE-full-v1/twitteraae_all"
AA = 0
HISPANIC = 1
OTHER = 2
WHITE = 3
def collect_data(database, race_id, output_file, threshold):
    data = []
    line_counter = 1
    with open(database, 'r') as f:
        searching = True
        while searching:
            line = f.readline()
            # print(line)
            comment = line.split("\t")[5]
            race_prob = line.split("\t")[6+race_id]
            
            is_rejection = line_counter >= 45e6 or np.random.random() < 1 / (45e6 - line_counter)
            if float(race_prob) > threshold and is_rejection:
                AA_prob = line.split("\t")[6+AA]
                HISPANIC_prob = line.split("\t")[6+HISPANIC]
                OTHER_prob = line.split("\t")[6+OTHER]
                WHITE_prob = line.split("\t")[6+WHITE]
                data.append((comment, AA_prob, HISPANIC_prob, OTHER_prob, WHITE_prob))
                if len(data) >= 1100:
                    searching = False
                    break
            line_counter += 1
            if line_counter % 50e6 == 0:
                print(f"Did not find enough data for {output_file} after {line_counter} lines")
                return
    with open(output_file, 'w') as f:
        for comment, AA_prob, HISPANIC_prob, OTHER_prob, WHITE_prob in data:
            f.write(f"{comment}\t{AA_prob}\t{HISPANIC_prob}\t{OTHER_prob}\t{WHITE_prob}")
            
    print(f"Found {len(data)} lines for {output_file} after {line_counter} lines")
    

def collect_data_all():
    inputs = []
    for threshold in [0.95]:
        for race_id, race_str in [(AA, 'AA'), (HISPANIC, 'HISPANIC'), (OTHER, 'OTHER'), (WHITE, 'WHITE')]:
            inputs.append((database,race_id,f'./race/{int(threshold*100)}/unlabeled-{race_str}-1100.csv',threshold))
            
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(collect_data, inputs)
        
def subsample(csv_file, output_file, sample_size):
    df = pd.read_csv(csv_file, sep="\t", header=None)
    df.columns = ['comment', 'AA_prob', 'HISPANIC_prob', 'OTHER_prob', 'WHITE_prob']
    df = df.sample(n=sample_size, random_state=1)
    df.to_csv(output_file, sep="\t", index=False, header=False)
    
def subsample_all():
    inputs = []
    for threshold in [0.95]:
        for race_id, race_str in [(AA, 'AA'), (HISPANIC, 'HISPANIC'), (OTHER, 'OTHER'), (WHITE, 'WHITE')]:
            inputs.append((
                f'./race/{int(threshold*100)}/unlabeled-{race_str}-1100.csv',
                f'./race/{int(threshold*100)}/labeled-{race_str}-100.csv',
                100))
            
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(subsample, inputs)
    
if __name__ == "__main__":
    subsample_all()
    
