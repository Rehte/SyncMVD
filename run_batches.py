import pandas as pd
import subprocess
import os

from tqdm import tqdm

max_hits = 2
style_prompt = "christmas style"

def main():
    # Read the CSV and extract the list of uids
    df = pd.read_csv('Objects.csv')
    uid_list = df['uid'].tolist()

    # Loop through the uid list and run the experiment for each uid
    for uid in tqdm(uid_list):
        config_path = f"objaverse/{uid}/config.yaml"
        
        if not os.path.exists(config_path):
            print(f"Config file missing: {config_path}")
            continue
        
        # Construct the command with max_hits
        command = f"python run_experiment.py --config {config_path} --max_hits {max_hits}"
        if style_prompt is not None:
            command += f" --style_prompt \"{style_prompt}\""
        
        # Run the command
        subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()
